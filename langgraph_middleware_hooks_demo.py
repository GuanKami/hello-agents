"""
实验名称：before_model / after_model 写入 Agent State
实验目标：观察 `create_agent` 内部 ReAct 循环中，`@before_model` 与 `@after_model`
    两个 hook 的真实触发时机，并验证"hook 返回的 dict 会被当作 State 更新合并进
    Agent State"这一语义与普通图节点完全一致。
解决的问题：之前的 `@dynamic_prompt` 实验只能影响"模型这一次看到什么提示词"，改动
    不落盘、State 里查不到。本实验要回答：Middleware 能不能真正修改工作流状态？
    如果能，改动能活多久、能在哪里被读到？
使用的 Agent 概念：
    - Middleware hook（before_model / after_model）
    - `create_agent` 内部隐式构建的 ReAct 循环（model -> after_model -> tools -> before_model）
    - State schema 合并（middleware 声明的 state_schema 会并入整图 State）
    - LangChain 的 `AgentState` 与 LangGraph 的 `MessagesState` 的区别
系统架构（本实验刻意保持最小，只观察 hook，不做别的）：

    START
      -> log_before_model.before_model      <- 自己写的 hook，循环入口，每轮都跑
      -> model                             <- create_agent 内部的模型节点
      -> log_after_model.after_model        <- 自己写的 hook，每轮出口
      -> 条件路由 ─┬─ 有 tool_calls -> tools -> 回到 before_model
                   └─ 无 tool_calls -> END

    State 中新增两个字段（由本实验声明）：
        model_call_count : before_model 每轮 +1，用于证明"触发次数 = 模型调用次数"
        last_tool_names  : after_model 记录本轮模型请求了哪些工具

实现方式：
    1. 定义 `MiddlewareState`，继承框架自带的 `AgentState`，新增上面两个字段。
    2. 用 `@before_model` / `@after_model` 装饰两个函数，各自返回 dict 作为 State patch。
    3. 通过 `create_agent(..., state_schema=MiddlewareState)` 把扩展字段并进整图 State。
    4. `main()` 发两种请求做对照：一种必然触发工具（问天气），一种不触发（寒暄）。

验证方式（两条路径，建议先跑零成本的那条）：
    路径 A（零 API 成本，确定性）：运行 `run_deterministic_selfcheck()`。
        它用假模型驱动一个完整的工具调用循环，期望看到：
            before_model 第 1 次，消息数 1
            after_model  看到 tool_calls = ['get_weather']
            before_model 第 2 次，消息数 3      <- 多了 AIMessage + ToolMessage
            after_model  看到 tool_calls = []
            最终 model_call_count == 2
        这是确定性验证，不依赖真实模型是否愿意调用工具。
    路径 B（真实模型，有 API 成本，属手动验证）：运行 `main()`。
        期望同样看到 hook 各触发 2 次（问天气）与 1 次（寒暄）。
        注意：免费/小模型不保证稳定调用工具，若它直接口头回答天气，会只触发 1 次，
        这是"模型不配合"而非 hook 配置错误。

学习总结（已实测确认，不是推测）：
    - `before_model` / `after_model` 会被编译成**真正的图节点**，节点名形如
      `{middleware名字}.before_model`。可以用 draw_mermaid() 看到它们。
    - 它们位于 ReAct 循环内部：`tools -> before_model -> model -> after_model`
      构成一轮迭代，所以"模型调用几次，两个 hook 就各触发几次"。
    - 它们返回的 dict 会被当成 State 更新合并进状态，且**在 invoke 返回后依然可读**。
      这一点是它们与 `@dynamic_prompt` 的本质区别——后者由 wrap_model_call 实现，
      只改这一次请求的 system message，不会写进 State。

已知限制：
    - Store 用的是 `InMemoryStore`，资料只在当前 Python 进程内有效，退出即丢失。
      （跨进程持久化应换 `SqliteStore`，见 AGENTS.md 12.2 节。）
    - 字段只在单次 `invoke` 内有效：没有接 checkpointer，每次 invoke 都是全新的
      State，所以计数会从 1 重新开始。这是刻意设计，不是缺陷。
    - 只注册了一个 before_model 和一个 after_model，没有验证多个 middleware 的
      组合顺序（多个 after_model 是**逆序**串联的，因为它是栈式包裹）。
    - 没有验证 `can_jump_to` 跳转、消息裁剪、模型切换等更高级的 hook 用法。

后续优化方向：
    - 把 Store 换成 SqliteStore，验证跨进程长期资料是否仍能被 hook 读到。
    - 增加第二个 before_model，观察多个同类型 hook 的执行顺序。
    - 尝试 `@before_model(can_jump_to=["end"])` + 返回 Command，为 Human-in-the-loop
      的 interrupt / 审批做铺垫。
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, after_model, before_model
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore

from tools import Context, get_weather

# 注意：必须在创建任何 ChatOpenAI 之前加载 .env。
# 这些模型对象在构造时就会去读 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL，
# 如果 load_dotenv() 晚于构造，模型会拿到 None 配置。
# 旧版本草稿把 load_dotenv() 放在 main() 里，且把模型建在模块顶层，
# 依赖 tools 模块的导入副作用才侥幸可用——那是脆弱的隐式依赖，已修正。
load_dotenv()


# ============================================================================
# 一、State：先搞清楚这个类从哪来、为什么要继承它
# ============================================================================
#
# 初学者最容易卡住的点：`AgentState` 我没有定义过，为什么可以直接继承？
#
# 答：`AgentState` 是 **LangChain 框架自带的类型**，不是本项目的类。
#     它从 `langchain.agents.middleware` 导入，定义位置在
#     `langchain/agents/middleware/types.py`，本身是一个 TypedDict：
#
#         class AgentState(TypedDict, Generic[ResponseT]):
#             messages: Required[...]              # 对话消息列表
#             jump_to: Optional[Literal[...]]      # 条件跳转目标
#             structured_response: ResponseT       # 结构化输出
#
#     你要做的只是"在它基础上扩展字段"，父类字段自动继承。
#
# 第二个容易混淆的点：为什么这里用 `AgentState`，而
# `langgraph_state_context_demo.py` 里用的是 `MessagesState`？
#
#     它们是**两个框架层各自定义的两个类，彼此之间没有继承关系**：
#
#         MessagesState  (来自 langgraph.graph)       只自带 messages
#         AgentState     (来自 langchain.agents...)   messages + jump_to + structured_response
#
#     `create_agent` 内部依赖 `jump_to` 实现 can_jump_to 跳转、依赖
#     `structured_response` 承载结构化输出，所以用 create_agent 时必须基于
#     `AgentState`。而你手写 StateGraph 时才用 `MessagesState`。
#
#     一句话：自己搭图时你定义 State；用高层 API 时框架定义 State，你只做扩展。
#
class MiddlewareState(AgentState):
    """
    Agent 工作流状态，在框架自带的 AgentState 上追加本实验需要的字段。

    字段首次注册时**还不存在**，所以代码里一律用 `state.get("字段", 默认值)` 读取，
    不能直接下标访问。原因：AgentState 继承了 TypedDict 的默认行为（total=True），
    这两个注解会被视作"必填"；而图在第一次进入 before_model 时 State 里
    确实还没有 model_call_count。用 `.get()` 提供默认值既安全，
    也准确表达了"这是运行时逐步填充的字段"。

    如果想让它更严谨，可以显式写 `model_call_count: NotRequired[int]`
    （需 from typing import NotRequired）。本实验保持注解最简，
    把"可缺省"这层语义交给 `.get()` 的默认值来表达。
    """

    # before_model 每轮 +1。用它来证明"hook 触发次数 == 模型调用次数"。
    model_call_count: int

    # after_model 记录本轮模型请求调用了哪些工具，用于观察模型行为。
    last_tool_names: list[str]


# ============================================================================
# 二、长期记忆 Store
# ============================================================================
def build_store() -> InMemoryStore:
    """
    创建本次程序运行期间共享的长期记忆 Store。

    为什么要把 Store 做成函数返回值，而不是模块级全局单例？
        因为 Store 是**运行时资源**，应该由"运行入口"决定用哪一个后端
        （现在用 InMemoryStore，将来换 SqliteStore 只改这里）。
        Node / hook 只通过 runtime.store 访问它，不关心它存在哪里——
        这就是依赖注入的好处。

    InMemoryStore 的生命周期限制：
        数据只活在当前 Python 进程内，进程一退出就全部丢失。
        所以本实验的 user_2 资料在每次运行时都会被重新写入。
    """
    store = InMemoryStore()

    store.put(
        ("users",),
        "user_2",
        {
            "name": "猪八戒",
            "language": "中文",
            "favorite_topics": ["美食"],
            "identity": "厨师",
        },
    )

    return store


# ============================================================================
# 三、两个 hook：本实验的核心
# ============================================================================
#
# 先解释装饰器语法，因为 `@before_model(state_schema=...)` 这种"带参数的装饰器"
# 很容易被误读成一条声明语句。
#
# `before_model` 本身就是一个普通函数，签名是：
#
#     def before_model(func=None, *, state_schema=None, tools=None,
#                      can_jump_to=None, name=None): ...
#
# 所以有两种写法：
#
#     写法 A（不带括号）：@before_model
#         直接把被装饰的函数当作 func 参数传进去。
#
#     写法 B（带括号）：  @before_model(state_schema=MiddlewareState)
#         先传入关键字参数、拿到一个装饰器，再作用到函数上。
#
# 两种写法都会返回一个 AgentMiddleware 实例，区别只是写法 B 额外声明了
# "我这个 hook 需要 State 里有 MiddlewareState 这些字段"。
#
#
# 那么 state_schema 到底做什么？——它只做**声明**，不做类型检查。
#
# `create_agent` 在编译时会收集所有 middleware 声明的 state_schema，
# 和基类 AgentState 求并集，得到整张图最终的 State 结构
# （源码位置：langchain/agents/factory.py，形如
#   state_schemas = [*(m.state_schema for m in middleware), base_state]
#   resolved_state_schema, _, _ = _resolve_schemas(state_schemas) ）。
#
# 注意 base_state 排在列表最后，意味着冲突时 create_agent(state_schema=...)
# 的声明优先。所以本实验把 state_schema 统一声明在 create_agent(...) 上
# （见 build_agent），下面两个装饰器因此不带参数。
#
# ⚠️ 必须记住的坑：如果自定义字段没有出现在合并后的 State schema 里，
#    你返回的 State patch 会被**静默丢弃，而且不报任何错**。
#    已实测：不声明 state_schema 时返回 {"model_call_count": 99}，
#    invoke 正常结束，但最终读到的是 None；声明后能正确读到 99。
#    所以调试"我的字段怎么不见了"时，第一个要检查的就是这个声明。
#
@before_model
def log_before_model(state: MiddlewareState, runtime) -> dict | None:
    """
    在每次调用模型**之前**执行。

    生命周期位置：它是 ReAct 循环的**入口**。`tools` 节点执行完会回到这里，
    所以只要模型还在请求工具，这个函数就会被反复触发。

    返回的 dict 会被合并进 State。这里刻意不用模块级全局变量来计数：
        - 同一次 invoke 内，State 在循环的各节点间传递，所以计数会累加到 2；
        - 不同 invoke 之间没有 checkpointer，State 是全新的，计数自动从 1 开始。
    用 State 承载计数器，本身就精确表达了"计数只在本次运行内有效"这一语义。
    """
    if not state["messages"]:
        return None

    # 从 State 读出上一轮的计数，+1 后写回。
    # 第一次进来时该字段还不存在，所以必须给默认值 0，否则 KeyError。
    count = state.get("model_call_count", 0) + 1

    print(
        f"[before_model] 第 {count} 次即将调用模型 | "
        f"当前消息数 = {len(state['messages'])} | "
        f"消息类型 = {[m.type for m in state['messages']]}"
    )

    # 返回 dict 即"我要把这些字段合并进 State"。
    # 注意：这不是 return 给调用方，而是 LangGraph 的 State 更新协议。
    return {"model_call_count": count}


@after_model
def log_after_model(state: MiddlewareState, runtime) -> dict | None:
    """
    在每次调用模型**之后**执行。

    生命周期位置：它是每轮迭代的**出口**。此刻 State 的最后一条消息就是模型
    刚刚产出、尚未被后续逻辑处理的 AIMessage，因此这里最适合做：
        - 观测模型行为（本实验）
        - 统计 token / 成本
        - 在消息落盘前拦截或改写（后续实验）

    与 @dynamic_prompt 的关键区别：
        @dynamic_prompt 由 wrap_model_call 实现，只能在"这次请求"的外层替换
        system message，改完就没了；而本函数可以返回 dict 写进 State，
        改动会被后续所有节点看到。
    """
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]

    # 只有 AIMessage 才带 tool_calls，其它类型用 getattr 兜底取 None。
    # 这里不能用 last_message.tool_calls 直接取——HumanMessage / ToolMessage
    # 没有这个属性，会 AttributeError。
    tool_calls = getattr(last_message, "tool_calls", None) or []
    tool_names = [call["name"] for call in tool_calls]

    print(
        f"[after_model] 模型产出 tool_calls = {tool_names} | "
        f"content = {last_message.content!r}"
    )

    # 额外打印第一条消息的类型，用它证明一件重要的事：
    # before_model / after_model **不能像 @dynamic_prompt 那样替换 system message**。
    # 如果真的替换了，这里会看到 'system'，而实测恒为 'human'。
    print(f"[after_model] messages[0].type = {state['messages'][0].type}（期望 human）")

    return {"last_tool_names": tool_names}


# ============================================================================
# 四、组装 Agent
# ============================================================================
def build_agent(store: InMemoryStore, model=None):
    """
    根据配置创建 Agent，但**不执行**用户请求。

    参数 model 允许注入：
        - 不传时，用 .env 配置的真实模型（走网络、有成本）；
        - 传入假模型时，可以零成本地确定性地验证 hook 行为。
    这种"把模型作为参数传进来"的写法就是依赖注入，和 build_store 的思路一致。

    为什么把 store 也作为参数而不是在函数内部新建？
        因为长期资料需要在"创建 Agent"和"之后直接读 Store 校验"之间共享同一个实例。
    """
    if model is None:
        model = ChatOpenAI(
            model=os.getenv("LLM_MODEL_ID"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )

    # create_agent 内部会隐式构建下面这张图（这就是它"隐藏"的 LangGraph 机制）：
    #
    #     START -> log_before_model.before_model -> model
    #           -> log_after_model.after_model
    #           -> 条件路由 ─┬─ tools -> 回到 before_model
    #                        └─ END
    #
    # 我们只声明了两个 hook，没有写任何 StateGraph 代码，
    # 但框架自动把它们编译成了图节点并接进了 ReAct 循环。
    # 想亲眼确认这张图，运行 main() 时会打印 draw_mermaid() 的结果。
    return create_agent(
        model=model,
        # 只保留 get_weather 一个工具：它不依赖 Store，行为完全可预测，
        # 这样"是否触发工具循环"只取决于模型的决策，不受 Store 内容干扰。
        tools=[get_weather],
        middleware=[log_before_model, log_after_model],
        system_prompt=(
            "你是一个助手。当用户询问天气时，必须调用 get_weather 工具获取信息，"
            "不要凭记忆直接回答。"
        ),
        context_schema=Context,
        # 在这里统一声明扩展后的 State。
        # 若删掉这一行，hook 返回的 model_call_count / last_tool_names
        # 会被静默丢弃（见上方注释中的实测结论）。
        state_schema=MiddlewareState,
        store=store,
    )


# ============================================================================
# 五、辅助函数
# ============================================================================
def print_messages(result: dict) -> None:
    """按时间顺序打印这次运行产生的全部消息，便于观察 ReAct 循环轨迹。"""
    for message in result["messages"]:
        message.pretty_print()


def print_state_summary(result: dict) -> None:
    """
    打印 hook 写进 State 的字段。

    这是本实验最关键的验收点：如果这两个值有内容，就证明
    "before_model / after_model 的返回值真的被当作 State 更新"，
    而不只是打印了一下。对比 @dynamic_prompt 的实验——那里
    无论怎么改提示词，最终 State 里都找不到任何痕迹。
    """
    print("=== 最终 State 摘要 ===")
    print(f"model_call_count = {result.get('model_call_count')}  （模型被调用的次数）")
    print(f"last_tool_names  = {result.get('last_tool_names')}  （最后一次模型请求的工具）")


# ============================================================================
# 六、路径 A：零 API 成本的确定性自检
# ============================================================================
class _ToolCallingFakeModel(GenericFakeChatModel):
    """
    可绑定工具的假模型，用于零成本、确定性地驱动一次完整工具循环。

    为什么要子类化？
        `GenericFakeChatModel` 虽然存在 bind_tools 属性，但 `create_agent` 的
        绑定路径会走到 ChatModel.bind_tools 的默认实现并抛 NotImplementedError。
        这里覆写成"返回自己"，让绑定变成空操作，同时按预设脚本依次产出消息。

    这是**测试替身**，只在自检里使用，不参与真实实验流程。
    """

    def bind_tools(self, tools, **kwargs):
        return self


def run_deterministic_selfcheck() -> None:
    """
    用假模型跑一遍完整的工具调用循环，验证 hook 的触发时机与 State 写入。

    这条路径不联网、不产生费用，可以反复运行。它验证的是 **hook 逻辑与图结构**，
    不是"真实模型是否愿意调用工具"——后者只能由路径 B 覆盖。

    期望输出（已实测）：
        [before_model] 第 1 次 ... 消息数 = 1
        [after_model] 模型产出 tool_calls = ['get_weather']
        [before_model] 第 2 次 ... 消息数 = 3      <- 多了 AIMessage 和 ToolMessage
        [after_model] 模型产出 tool_calls = []
        最终 model_call_count = 2
    """
    print("\n" + "=" * 70)
    print("路径 A：确定性自检（假模型，零 API 成本）")
    print("=" * 70)

    # 假模型的消息队列是"脚本"：第一次产出带 tool_calls 的 AIMessage，
    # 第二次产出纯文本。这正好模拟"先调工具、再总结"的 ReAct 行为。
    scripted_messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"city": "北京"}, "id": "call_1"}
            ],
        ),
        AIMessage(content="北京今天晴天。"),
    ]

    agent = build_agent(
        build_store(),
        model=_ToolCallingFakeModel(messages=iter(scripted_messages)),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]},
        context=Context(user_id="user_2", authority="user"),
    )

    print_state_summary(result)

    count = result.get("model_call_count")
    assert count == 2, f"期望 model_call_count == 2，实际 {count}"
    print("\n[PASS] 自检通过：hook 触发 2 次，且返回值确实写入了 State。")


# ============================================================================
# 七、路径 B：真实模型运行（有 API 成本，属手动验证）
# ============================================================================
def main() -> None:
    """用真实模型演示 hook 行为。这是手动实验，会产生外部 LLM API 调用与费用。"""
    store = build_store()
    agent = build_agent(store)

    # 先打印静态图结构。这是零成本的，且能直观看到两个 hook 被编译成了图节点。
    # 注意：draw_mermaid() 只是静态图描述，不代表真实执行轨迹。
    print("=== create_agent 生成的图结构（静态） ===")
    print(agent.get_graph().draw_mermaid())

    # ---- 请求 1：会触发工具调用，期望两个 hook 各触发 2 次 ----
    print("\n=== 请求 1：问天气（期望触发工具循环） ===")
    result_01 = agent.invoke(
        {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]},
        context=Context(user_id="user_2", authority="user"),
    )
    print_messages(result_01)
    print_state_summary(result_01)

    # ---- 请求 2：不需要工具，期望两个 hook 各只触发 1 次 ----
    # 这次调用也验证了：没有 checkpointer 时，上一次的计数不会带过来，
    # model_call_count 会重新从 1 开始。
    print("\n=== 请求 2：寒暄（期望不触发工具） ===")
    result_02 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}]},
        context=Context(user_id="user_2", authority="user"),
    )
    print_messages(result_02)
    print_state_summary(result_02)


if __name__ == "__main__":
    # 默认跑零成本自检。想跑真实模型时，把下面这行改成 main()。
    run_deterministic_selfcheck()
