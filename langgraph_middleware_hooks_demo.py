"""
实验名称：before_model / after_model 写入 Agent State
实验目标：
    1. 观察 @before_model 与 @after_model 在 create_agent 内部 ReAct 循环中的
       真实触发时机，证明"hook 触发次数 == 模型调用次数"。
    2. 验证"hook 返回的 dict 会被当作 State 更新合并进 Agent State"这一语义，
       与普通图节点的返回值语义完全一致。

解决的问题：
    上一个实验 @dynamic_prompt 只能影响"模型这一次看到什么提示词"，
    改动不落盘、State 里查不到、下一次调用就没了。
    本实验要回答一个更根本的问题：
        Middleware 能不能真正修改工作流状态？改动能活多久？在哪里能被读到？

使用的 Agent 概念：
    - Middleware hook（before_model / after_model）
    - create_agent 内部隐式构建的 ReAct 循环
    - State schema 声明与合并（middleware 声明的 state_schema 会并入整图 State）
    - LangChain 的 AgentState（来自 langchain.agents.middleware）与
      LangGraph 的 MessagesState（来自 langgraph.graph）的区别

系统架构（本实验刻意保持最小，只观察 hook，不做别的）：

    START
      -> count_model_calls.before_model      <- 自己写的 hook，循环入口，每轮都跑
      -> model                               <- create_agent 内部的模型节点
      -> inspect_model_output.after_model    <- 自己写的 hook，每轮出口
      -> 条件路由 ─┬─ 有 tool_calls -> tools -> 回到 before_model
                   └─ 无 tool_calls -> END

    State 中新增两个字段（由本实验声明）：
        model_call_count : before_model 每轮 +1，用于证明"触发次数 = 模型调用次数"
        last_tool_names  : after_model 记录本轮模型请求了哪些工具

实现方式：
    1. 定义 CallCountState，继承框架自带的 AgentState，新增上面两个字段。
    2. 用 @before_model / @after_model 装饰两个函数，各自返回 dict 作为 State patch。
    3. 通过 create_agent(..., state_schema=CallCountState) 把扩展字段并进整图 State。
    4. main() 发两种请求做对照：一种会触发工具（问天气），一种不触发（寒暄）。

验证方式（手动实验，会产生真实 LLM API 调用与费用）：
    运行 main()，肉眼核对以下三点，全部符合即为通过：

    请求 A（"你好"，寒暄，不触发工具）：
        [before_model] 第 1 次 ... 当前 State 消息数 = 1
        [after_model]  模型产出 tool_calls = []
        之后没有第二次 —— 因为模型直接回答了，循环结束。

    请求 B（"北京今天天气怎么样？"，触发工具）：
        [before_model] 第 1 次 ... 当前 State 消息数 = 1
        [after_model]  模型产出 tool_calls = ['get_weather']
        [before_model] 第 2 次 ... 当前 State 消息数 = 3   <- 多了 2 条
        [after_model]  模型产出 tool_calls = []
        最后看"最终 State 摘要"，model_call_count 应为 2。

    为什么第 2 次是 3 而不是 2：
        1 条 HumanMessage
      + 1 条"模型决定调用工具"的 AIMessage（带 tool_calls）
      + 1 条"工具执行结果"的 ToolMessage
      = 3 条。
        这 2 条新增消息正是 ReAct 一轮完整的"行动 + 观察"。它们必须被 append 进
        state["messages"]，因为 LLM 是无状态的：第 2 次调用时如果不把完整历史
        重新发过去，模型就不知道"自己上一轮已经查过天气、结果已经拿到"，
        于是会再查一次 -> 又一轮 -> 无限循环。
        注意这 3 条**不包含**模型第 2 次产出的最终回答，因为本 hook 是在
        "即将调用模型之前"读取 State，那条回答此时还没被生成。

学习总结（已实测确认，不是推测）：
    1. before_model / after_model 会被编译成**真正的图节点**，节点名形如
       {middleware名字}.before_model。用 agent.get_graph().draw_mermaid() 可以看到。
    2. 它们位于 ReAct 循环内部：tools -> before_model -> model -> after_model
       构成一轮迭代。所以"模型调用几次，两个 hook 就各触发几次"。
    3. 它们返回的 dict 会被当成 State 更新合并进状态，且**在 invoke 返回后依然可读**。
       这一点是它们与 @dynamic_prompt 的本质区别 —— 后者由 wrap_model_call 实现，
       只改这一次请求的 system message，不会写进 State。

已知限制：
    - Store 用的是 InMemoryStore，资料只在当前 Python 进程内有效，退出即丢失。
      （跨进程持久化应换 SqliteStore，见 AGENTS.md 12.2 节。）
    - 字段只在单次 invoke 内有效：没有接 checkpointer，每次 invoke 都是全新的
      State，所以 main() 里第二次请求的计数会从 1 重新开始，而不是接着上一次。
      这是刻意设计，不是缺陷。
    - 只注册了一个 before_model 和一个 after_model，没有验证多个 middleware 的
      组合顺序（多个 after_model 是**逆序**串联的，因为它是栈式包裹）。
    - 没有验证 can_jump_to 跳转、消息裁剪、模型切换等更高级的 hook 用法。
    - 真实模型路径依赖模型"愿意调用工具"。免费/小模型可能直接凭记忆回答天气，
      那样只会触发 1 次。这属于"模型不配合"，不是 hook 配置错误。
    - 本实验刻意只用真实模型验证，不引入假模型/Mock：每次验证都会产生真实
      API 调用与费用，且结果受模型当时行为影响，不可完全复现。
      这是有意选择（保持实验贴近真实运行），代价是验证不是确定性的。

后续优化方向：
    - 把 Store 换成 SqliteStore，验证跨进程长期资料是否仍能被 hook 读到。
    - 增加第二个 before_model，观察多个同类型 hook 的执行顺序。
    - 尝试 @before_model(can_jump_to=["end"]) + 返回 Command，
      为 Human-in-the-loop 的 interrupt / 审批做铺垫。
"""
import os

from typing import NotRequired
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime

from tools import Context, get_weather, get_user_info, save_user_info

# ============================================================================
# 一、模型与 .env 的加载顺序（这里有个容易踩的坑）
# ============================================================================
#
# 必须先 load_dotenv()，再构造 ChatOpenAI。
#
# 为什么：ChatOpenAI 在**构造时**就会读取 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL
# 这三个值。如果 load_dotenv() 晚于构造，os.getenv() 会返回 None，
# 模型就带着空配置被创建出来。
#
# 旧版本把 load_dotenv() 写在 main() 里、却把 ChatOpenAI 建在模块顶层，
# 它之所以"看起来能跑"，是因为下面这行 `from tools import ...` 先执行了，
# 而 tools.py 顶部自带 load_dotenv()，等于替本模块把 .env 加载进了 os.environ。
# 那是**隐式副作用**，不是本模块的正确性保证：一旦调整 import 顺序就会静默失效。
# 现在把 load_dotenv() 显式提到 import 之后、构造模型之前，去掉这个隐患。
load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

# ============================================================================
# 二、长期记忆 Store
# ============================================================================
def build_store() -> InMemoryStore:
    """
    创建本次程序运行期间共享的长期记忆 Store。

    为什么把 Store 做成函数返回值，而不是模块级全局单例？
        因为 Store 是**运行时资源**，应该由"运行入口"决定用哪一个后端
        （现在用 InMemoryStore，将来换 SqliteStore 只改这里）。
        Node / hook 只通过 runtime.store 访问它，不关心它存在哪里 ——
        这就是依赖注入，和后端里"Repository 接口不关心底层是 MySQL 还是 PG"同理。

    InMemoryStore 的生命周期限制：
        数据只活在当前 Python 进程内，进程一退出就全部丢失。
        所以本实验的资料在每次运行时都会被重新写入。

    注意：本实验的两个 hook 都没有用到 Store，这里保留它是为了与
    @dynamic_prompt 实验保持同样的骨架，方便对照。
    """
    store = InMemoryStore()

    # namespace ("users",) + key "user_1" 构成长期记忆的精确寻址路径。
    # 用精确 key 而不是语义 search()，因为姓名/语言这类是结构化事实，
    # 需要 100% 准确，不能接受向量检索的近似排序。
    store.put(
        ("users",),
        "user_1",
        {
            "name": "派大星",
            "language": "中文",
            "favorite_topics": ["LangGraph", "LangChain"],
            "identity": "从业者",
        },
    )

    store.put(
        ("users",),
        "user_2",
        {
            "name": "猪八戒",
            "language": "English",
            "favorite_topics": ["美食"],
            "identity": "厨师",
        },
    )

    return store


# ============================================================================
# 三、State：先搞清楚这个类从哪来、为什么要继承它
# ============================================================================
#
# 初学者最容易卡住的点：AgentState 我没有定义过，为什么可以直接继承？
#
# 答：AgentState 是 **LangChain 框架自带的类型**，不是本项目的类。
#     它从 langchain.agents.middleware 导入，定义在
#     langchain/agents/middleware/types.py，本身是一个 TypedDict：
#
#         class AgentState(TypedDict, Generic[ResponseT]):
#             messages: Required[...]           # 对话消息列表
#             jump_to: Optional[Literal[...]]   # 条件跳转目标
#             structured_response: ResponseT    # 结构化输出
#
#     你要做的只是"在它基础上扩展字段"，父类字段自动继承。
#
# 第二个容易混淆的点：为什么这里用 AgentState，而
# langgraph_state_context_demo.py 里用的是 MessagesState？
#
#     它们是**两个框架层各自定义的两个类，彼此之间没有继承关系**：
#
#         MessagesState  (来自 langgraph.graph)        只自带 messages
#         AgentState     (来自 langchain.agents...)    messages + jump_to + structured_response
#
#     create_agent 内部依赖 jump_to 实现 can_jump_to 跳转、依赖
#     structured_response 承载结构化输出，所以用 create_agent 时必须基于
#     AgentState。而你手写 StateGraph 时才用 MessagesState。
#
#     一句话：自己搭图时你定义 State；用高层 API 时框架定义 State，你只做扩展。
#
class CallCountState(AgentState):
    """
    Agent 工作流状态，在框架自带的 AgentState 上追加本实验需要的字段。

    为什么用 NotRequired：
        这些字段在第一次进入 before_model 时**还不存在**，
        是运行时由 hook 逐步填充的。标成 NotRequired 既表达了
        "可以缺省"，也提醒读取时要用 state.get("字段", 默认值)。

    ⚠️ 这里必须解释一个静默陷阱（本实验最重要的坑）：
        state_schema 只做**声明**，不做类型检查。
        如果某个自定义字段没有出现在合并后的 State schema 里，
        hook 返回的对应 patch 会被**静默丢弃，而且不报任何错**。

        已实测：不声明 state_schema 时返回 {"model_call_count": 99}，
        invoke 正常结束、不抛异常，但最终读到的是 None；声明后能正确读到 99。

        所以当你发现"我的字段怎么不见了"，第一个要检查的就是：
        这个字段有没有出现在整图合并后的 State schema 里。
    """

    # before_model 每轮 +1。用它来证明"hook 触发次数 == 模型调用次数"。
    model_call_count: NotRequired[int]

    # after_model 记录本轮模型请求调用了哪些工具，用于观察模型行为。
    last_tool_names: NotRequired[list[str]]

    # 证明确实写进了 State 的占位字段。
    # 当前两个 hook 都没有写它，保留是为了让你可以亲手加一行
    # return {"middleware_note": "..."} 验证写入行为。
    middleware_note: NotRequired[str]


# ============================================================================
# 四、两个 hook：本实验的核心
# ============================================================================
#
# 先解释装饰器语法，因为 @before_model(state_schema=...) 这种"带参数的装饰器"
# 很容易被误读成一条声明语句。
#
# before_model 本身就是一个普通函数，签名是：
#
#     def before_model(func=None, *, state_schema=None, tools=None,
#                      can_jump_to=None, name=None): ...
#
# 所以有两种写法：
#
#     写法 A（不带括号）：@before_model
#         直接把被装饰的函数当作 func 参数传进去。
#
#     写法 B（带括号）：  @before_model(state_schema=CallCountState)
#         先传入关键字参数、拿到一个装饰器，再作用到函数上。
#
# 两种写法都会返回一个 AgentMiddleware 实例，区别只是写法 B 额外声明了
# "我这个 hook 需要 State 里有 CallCountState 这些字段"。
#
# 本文件采用**写法 B**，在两个装饰器上各自声明 state_schema，
# 同时也在 create_agent(...) 里显式传了 state_schema=CallCountState（见 build_agent）。
# 两处都写是冗余但安全的：create_agent 编译时会收集所有 middleware 声明的
# state_schema 与 base_state 求并集（源码位置 langchain/agents/factory.py，
# 形如 state_schemas = [*(m.state_schema for m in middleware), base_state]），
# 两处声明同一个类不会冲突。
#
# 额外提醒：hook 生成的 middleware 类名默认取**函数名**。
# 如果将来再加一个 before_model，且两个函数恰好同名，
# create_agent 会抛 AssertionError: Please remove duplicate middleware instances.
# 那种情况需要显式传 name="..." 区分。
#
@before_model(state_schema=CallCountState)
def count_model_calls(state: CallCountState, runtime: Runtime) -> dict | None:
    """
    在每次调用模型**之前**执行。

    生命周期位置：它是 ReAct 循环的**入口**。tools 节点执行完会回到这里，
    所以只要模型还在请求工具，这个函数就会被反复触发。

    返回的 dict 会被合并进 State。这里刻意不用模块级全局变量来计数：
        - 同一次 invoke 内，State 在循环的各节点间传递，所以计数会累加到 2；
        - 不同 invoke 之间没有 checkpointer，State 是全新的，计数自动从 1 开始。
    用 State 承载计数器，本身就精确表达了"计数只在本次运行内有效"这一语义。

    参数 runtime 本实验没有用到，但保留它是为了展示 hook 的签名形状：
    所有 middleware hook 都接收 (state, runtime)，需要读身份/权限/Store 时
    就用 runtime.context 和 runtime.store，和 @dynamic_prompt 里的
    request.runtime 是同一个东西。
    """
    # 从 State 读出上一轮的计数，+1 后写回。
    # 第一次进来时该字段还不存在（NotRequired），所以必须给默认值 0，
    # 否则会 KeyError。
    n = state.get("model_call_count", 0) + 1

    # 打印消息数，用来观察 ReAct 循环的累积过程：
    # 不触发工具时恒为 1；触发工具后第二轮变成 3（+AIMessage +ToolMessage）。
    print(f"[before_model] 第 {n} 次调用模型, "
          f"当前 State 消息数 = {len(state['messages'])}")

    # ← 关键:返回 dict 写入 State
    # 注意：这不是"return 给调用方"，而是 LangGraph 的 State 更新协议 ——
    # 框架会把这个 dict 按 reducer 规则合并进 State。messages 字段的 reducer
    # 是 append（追加），而 model_call_count 这种标量字段默认是覆盖。
    return {"model_call_count": n}


@after_model(state_schema=CallCountState)
def inspect_model_output(state: CallCountState, runtime: Runtime) -> dict | None:
    """
    在每次调用模型**之后**执行。

    生命周期位置：它是每轮迭代的**出口**。此刻 State 的最后一条消息就是模型
    刚刚产出、尚未被后续逻辑处理的 AIMessage，因此这里最适合做：
        - 观测模型行为（本实验）
        - 统计 token / 成本
        - 在消息落盘前拦截或改写（后续实验，本次不做）

    与 @dynamic_prompt 的关键区别：
        @dynamic_prompt 由 wrap_model_call 实现，只能在"这次请求"的外层替换
        system message，改完就没了；而本函数可以返回 dict 写进 State，
        改动会被后续所有节点看到。

    为什么能看到 tool_calls：
        模型"决定要调工具"这个动作，本身就体现为它产出的 AIMessage 上带了
        tool_calls 字段。所以这里读到非空 tool_calls，就说明接下来
        条件路由会走向 tools 节点，并再回到 before_model —— 这就是"第 2 次触发"的来源。
    """
    # 只有 AIMessage 才带 tool_calls，HumanMessage / ToolMessage 没有这个属性。
    # 所以不能用 last.tool_calls 直接取（会 AttributeError），
    # 要用 getattr 兜底取 None，再用 or [] 归一成列表。
    last = state["messages"][-1]  # 获取最新的模型输出消息
    calls = getattr(last, "tool_calls", None) or []

    print(f"[after_model] 模型产出 tool_calls = {[c['name'] for c in calls]}")

    # 把本轮请求的工具名记进 State。
    # 同一个 invoke 内会被后续轮次覆盖，所以最终留下的是**最后一次**模型调用的结果。
    # 这也解释了为什么请求 B 结束时 last_tool_names == []（最后一轮不再需要工具）。
    return {"last_tool_names": [c["name"] for c in calls]}


# ============================================================================
# 五、组装 Agent
# ============================================================================
def build_agent(store: InMemoryStore):
    """
    根据配置创建 Agent，但**不执行**用户请求。

    为什么把 store 作为参数而不是在函数内部新建？
        因为长期资料需要在"创建 Agent"和"之后直接读 Store 校验"之间共享同一个实例。

    注意这里又新建了一个 ChatOpenAI，而不是复用模块顶层的 llm：
        这是一种**依赖注入**写法 —— 谁需要模型谁自己声明配置。
        好处是 build_agent 不依赖模块级全局状态，将来要在同一个进程里
        为不同请求构造不同配置的 Agent，只改这个函数即可。
        代价是当前存在两个配置相同的 llm 对象，略有冗余。
    """
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    # create_agent 内部会隐式构建下面这张图（这就是它"隐藏"的 LangGraph 机制）：
    #
    #     START -> count_model_calls.before_model -> model
    #           -> inspect_model_output.after_model
    #           -> 条件路由 ─┬─ tools -> 回到 before_model
    #                        └─ END
    #
    # 我们只声明了两个 hook，没有写任何 StateGraph 代码，
    # 但框架自动把它们编译成了图节点并接进了 ReAct 循环。
    # 想亲眼确认这张图，可以看 main() 里打印的 draw_mermaid() 结果。
    return create_agent(
        model=llm,
        # 工具集合：get_weather 用于触发工具循环；另外两个依赖 Store，
        # 保留它们是为了让这个 Agent 的骨架和 @dynamic_prompt 实验一致。
        tools=[get_weather, get_user_info, save_user_info],
        middleware=[count_model_calls, inspect_model_output],
        system_prompt="You are a helpful assistant",
        context_schema=Context,
        # 在这里声明扩展后的 State。
        # 若删掉这一行（且装饰器上也不声明），hook 返回的
        # model_call_count / last_tool_names 会被**静默丢弃**（见 CallCountState 注释）。
        state_schema=CallCountState,
        store=store,
    )


# ============================================================================
# 六、辅助函数
# ============================================================================
def print_messages(result: dict) -> None:
    """按时间顺序打印这次运行产生的全部消息，便于观察 ReAct 循环轨迹。"""
    for message in result["messages"]:
        message.pretty_print()


def print_state_summary(result: dict) -> None:
    """
    打印 hook 写进 State 的字段。

    这是本实验**最关键的验收点**：
        如果这两个值有内容，就证明"before_model / after_model 的返回值
        真的被当作 State 更新合并了"，而不只是打印了一下。
        对比 @dynamic_prompt 的实验 —— 那里无论怎么改提示词，
        最终 State 里都找不到任何痕迹。
    """
    print("=== 最终 State 摘要 ===")
    print(f"model_call_count = {result.get('model_call_count')}  （模型被调用的次数）")
    print(f"last_tool_names  = {result.get('last_tool_names')}  （最后一次模型请求的工具）")


# ============================================================================
# 七、运行入口
# ============================================================================
def main() -> None:
    """
    用真实模型演示 hook 行为。

    ⚠️ 这是**手动实验**，会产生外部 LLM API 调用与费用。
       运行前需确认 .env 与网络可用。仓库当前没有测试框架，
       所以这里的结果属于手动验证，不是自动化测试。
    """
    # .env 已在模块顶部加载（见文件开头第一段的解释），这里不再重复调用。

    # 同一个程序运行期间只创建一份 Store 和 Agent。
    store = build_store()
    agent = build_agent(store)

    # 先打印静态图结构。这是零成本的，且能直观看到两个 hook 被编译成了图节点。
    # 注意：draw_mermaid() 只是静态图描述，不代表真实执行轨迹。
    print("=== create_agent 生成的图结构（静态） ===")
    print(agent.get_graph().draw_mermaid())

    # ---- 请求 1：不触发工具，期望两个 hook 各只触发 1 次 ----
    print("\n=== 请求 1：寒暄（期望不触发工具循环） ===")
    result01 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "你好",
                }
            ]
        },
        context=Context(
            user_id="user_1",
            authority="user",
        )
    )
    print_messages(result01)
    print_state_summary(result01)

    print("\n========= 分割线 =========")

    # ---- 请求 2：触发工具，期望两个 hook 各触发 2 次 ----
    # 这次调用也验证了：没有 checkpointer 时，上一次的计数不会带过来，
    # model_call_count 会重新从 1 开始。
    print("\n=== 请求 2：问天气（期望触发工具循环） ===")
    result02 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "北京今天天气怎么样？",
                }
            ]
        },
        context=Context(
            user_id="user_2",
            authority="user",
        )
    )
    print_messages(result02)
    print_state_summary(result02)


if __name__ == "__main__":
    main()
