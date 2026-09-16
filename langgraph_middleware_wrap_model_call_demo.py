import os

from typing import NotRequired
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, wrap_model_call, AgentState, ModelRequest, ModelResponse
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime
from langchain.messages import AIMessage

from tools import Context, get_weather, get_user_info, save_user_info


# --- 一、模型与 .env 的加载顺序 ---------------------------------------------
# 必须先 load_dotenv() 再构造 ChatOpenAI：ChatOpenAI **在构造时**就读取 LLM_MODEL_ID /
# LLM_API_KEY / LLM_BASE_URL，晚了 os.getenv() 会返回 None，模型带着空配置被创建出来。
#
# 旧版本把 load_dotenv() 写在 main() 里，却靠 tools.py 顶部的 load_dotenv() 顺带生效。
# 那是隐式副作用 —— 一改 import 顺序就静默失效。现在显式放在构造之前。
load_dotenv()

# 低费率模型
basic_model = ChatOpenAI(
    model=os.getenv("BASIC_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

# 高费率模型
advanced_model = ChatOpenAI(
    model=os.getenv("ADVANCED_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

# --- 二、长期记忆 Store -----------------------------------------------------
def build_store() -> InMemoryStore:
    """
    创建本次程序运行期间共享的长期记忆 Store。

    做成函数返回值（依赖注入）而不是模块级全局单例，是为了让"运行入口"决定用哪个后端；
    hook / Node 只通过 runtime.store 访问它，不关心数据存在哪里。换 SqliteStore 只改这里。

    InMemoryStore 的数据只活在当前 Python 进程内，进程退出即全部丢失。
    本实验两个 hook 都没用到 Store，保留它是为了与 @dynamic_prompt 实验骨架一致。
    """
    store = InMemoryStore()

    # namespace ("users",) + key "user_1" 构成长期记忆的精确寻址路径。
    # 用精确 key 而不是语义 search()：姓名/语言是结构化事实，需要 100% 准确，
    # 不能接受向量检索的近似排序。
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


# --- 三、State：AgentState 从哪来、为什么要继承它 ---------------------------
# AgentState 是 **LangChain 框架自带的 TypedDict**（langchain.agents.middleware），自带
# messages / jump_to / structured_response，你只是扩展它，父类字段自动继承。
#
# 它和 MessagesState（来自 langgraph.graph）**没有继承关系**，是两个框架层各自的类：
#     MessagesState   只自带 messages                            -> 手写 StateGraph 时用
#     AgentState      messages + jump_to + structured_response   -> create_agent 时用
# create_agent 内部靠 jump_to 实现 can_jump_to、靠 structured_response 承载结构化输出，
# 所以用高层 API 时必须基于 AgentState。
class CallCountState(AgentState):
    """
    在框架自带的 AgentState 上追加本实验需要的字段。

    用 NotRequired 是因为这些字段在第一次进入 before_model 时还不存在，由 hook 在运行时
    逐步填充；读取时要配 state.get("字段", 默认值)。

    ⚠️ 本实验最重要的坑（静默陷阱）：state_schema 只做**声明**，不做类型检查。自定义字段
    若没出现在合并后的 State schema 里，hook 返回的 patch 会被**静默丢弃且不报错**。
    已实测：不声明 state_schema 时返回 {"model_call_count": 99}，invoke 正常结束、不抛
    异常，但最终读到 None；声明后读到 99。
    所以"我的字段怎么不见了"的第一个检查点，是它有没有进合并后的 State schema。
    """

    # before_model 每轮 +1，用于证明"hook 触发次数 == 模型调用次数"。
    model_call_count: NotRequired[int]

    # after_model 记录本轮模型请求了哪些工具。
    last_tool_names: NotRequired[list[str]]

    # 占位字段，当前两个 hook 都没写它。保留是为了让你能亲手加一行
    # return {"middleware_note": "..."} 验证写入行为。
    middleware_note: NotRequired[str]


# --- 四、两个 hook：本实验的核心 -------------------------------------------
# @before_model(state_schema=...) 是"带参数的装饰器"：before_model 本身是普通函数
# （def before_model(func=None, *, state_schema=None, tools=None, can_jump_to=None, name=None)），
# 先收关键字参数、返回装饰器，再作用到函数上；不带括号的 @before_model 也合法，
# 区别只是没声明 state_schema。
#
# 两个装饰器都声明了 state_schema、create_agent 里又传了一次：冗余但安全 —— 编译时会把
# 各 middleware 声明的 schema 与 base_state 求并集，重复声明同一个类不冲突。
#
# 注意 hook 生成的 middleware 类名默认取**函数名**；若将来两个 hook 恰好同名，会抛
# AssertionError: Please remove duplicate middleware instances.，需显式传 name="..."。
@before_model(state_schema=CallCountState)
def count_model_calls(state: CallCountState, runtime: Runtime) -> dict | None:
    """
    每次调用模型**之前**执行，是 ReAct 循环的**入口**：tools 节点执行完会回到这里，
    所以只要模型还在请求工具，它就会被反复触发。

    计数刻意放在 State 而不是模块级全局变量：同一次 invoke 内各节点共享 State，所以能
    累加到 2；不同 invoke 之间没有 checkpointer、State 全新，计数自动从 1 开始 ——
    这正好表达了"计数只在本次运行内有效"这一语义。

    runtime 参数本实验没用到，保留是为了展示签名形状：需要身份 / 权限 / Store 时用
    runtime.context 与 runtime.store，与 @dynamic_prompt 里的 request.runtime 是同一个东西。
    """
    # 第一次进来时该字段还不存在（NotRequired），必须给默认值 0，否则 KeyError。
    n = state.get("model_call_count", 0) + 1

    # 消息数用来看 ReAct 循环的累积：不触发工具时恒为 1，触发工具后第二轮变成 3。
    print(f"[before_model] 第 {n} 次调用模型, "
          f"当前 State 消息数 = {len(state['messages'])}")

    # 返回 dict 不是"返回给调用方"，而是 LangGraph 的 State 更新协议：框架按 reducer 规则
    # 把它合并进 State。messages 的 reducer 是 append（追加），标量字段默认是覆盖。
    return {"model_call_count": n}


@after_model(state_schema=CallCountState)
def inspect_model_output(state: CallCountState, runtime: Runtime) -> dict | None:
    """
    每次调用模型**之后**执行，是每轮迭代的**出口**。此刻 State 的最后一条就是模型刚刚
    产出、尚未被后续处理的 AIMessage，适合观测模型行为（本实验）、统计 token / 成本，
    或在消息落盘前拦截改写（后续实验）。

    与 @dynamic_prompt 的关键区别：后者由 wrap_model_call 实现，只在"这次请求"的外层
    替换 system message，改完就没了；本函数可以返回 dict 写进 State，改动会被后续所有
    节点看到。

    为什么能看到 tool_calls：模型"决定调工具"这个动作本身就体现为 AIMessage 上带了
    tool_calls 字段。这里读到非空 tool_calls，就说明条件路由接下来会走向 tools 节点，
    并再回到 before_model —— 这就是"第 2 次触发"的来源。
    """
    # 只有 AIMessage 带 tool_calls，HumanMessage / ToolMessage 没有这个属性。所以不能用
    # last.tool_calls 直接取（会 AttributeError），要用 getattr 兜底再 or [] 归一成列表。
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []

    print(f"[after_model] 模型产出 tool_calls = {[c['name'] for c in calls]}")

    # 该字段是标量、reducer 为覆盖：同一 invoke 内会被后续轮次覆盖，最终留下的是**最后
    # 一次**模型调用的结果。这也解释了请求 B 结束时为什么是 []（最后一轮不再需要工具）。
    return {"last_tool_names": [c["name"] for c in calls]}

# ① 观察层：打印 + 真的调用模型
@wrap_model_call
def observe_model_call(request: ModelRequest, handler) -> ModelResponse:

    # state 只读
    print(f"[wrap] 消息数 = {len(request.state['messages'])}")
    print(f"[wrap] system_message = {request.system_message}")
    response = handler(request)

    # <- 真正的网络请求
    ai_msg = response.result[0]
    print(f"[wrap] 下游响应的 tool_calls = {[tc['name'] for tc in (ai_msg.tool_calls or [])]}")
    print(f"[wrap] response_metadata = {ai_msg.response_metadata}")
    print(f"[wrap] usage_metadata = {ai_msg.usage_metadata}")
    print(f"[wrap] message_id = {ai_msg.id}")

    return response


# ② 短路层：命中就返回，handler 一次都不调
@wrap_model_call
def local_cache_shortcut(request: ModelRequest, handler) -> ModelResponse:
    """
    命中“本地缓存”关键词时，直接返回本地构造的响应。

    关键点：
    命中缓存后不执行 handler(request)。

    在当前 middleware 注册顺序下，handler 的下游是：
        provider_call_probe
            -> 真实模型

    因此这里不调用 handler，就会跳过：
        provider_call_probe
        真实模型
        后续模型产生的 tool_calls
    """
    latest_content = str(request.messages[-1].content)

    if "本地缓存" in latest_content:
        print("[cache] 命中本地缓存，不调用下游 handler")

        return ModelResponse(
            result=[
                AIMessage(
                    content="[来自本地缓存，未调用模型]",
                    additional_kwargs={
                        "response_metadata": "local_cache",
                    },
                )
            ]
        )

    print("[cache] 未命中，继续调用下游 handler")
    
    return handler(request)

# 仅用于本次单进程教学实验。
# 它不是 Agent State，也不是生产环境中的监控方案。
provider_call_count = {"value": 0}

@wrap_model_call
def provider_call_probe(request: ModelRequest, handler) -> ModelResponse:
    """
    真实模型调用探针。

    这个 wrapper 被放在本地短路层的内侧。
    只有 local_cache_shortcut 没有短路、继续调用 handler 时，
    才会进入这里。

    因此：
    - 普通请求：会执行这个探针；
    - 本地缓存命中：不会执行这个探针。
    """
    provider_call_count["value"] += 1

    print(
        "[provider_probe] 进入真实模型调用层, "
        f"累计次数 = {provider_call_count['value']}"
    )

    # 这里的 handler 通常已经是最内层的真实模型调用。
    response = handler(request)
    return response

# --- 五、组装 Agent ---------------------------------------------------------
def build_agent(store: InMemoryStore):
    """
    根据配置创建 Agent，但**不执行**用户请求.

    store 作为参数传入，是为了让"创建 Agent"和"之后直接读 Store 校验"共享同一个实例。

    模型直接用模块顶层的 advanced_model：它同时是 dynamic_model_selection 的默认选择，
    "默认模型"和"可被中间件替换的模型"共用同一个对象，避免出现两份配置不一致的实例。
    """

    # create_agent 内部会隐式构建下面这张图（这就是它"隐藏"的 LangGraph 机制）：
    #
    #     START -> count_model_calls.before_model -> model
    #           -> inspect_model_output.after_model
    #           -> 条件路由 ─┬─ tools -> 回到 before_model
    #                        └─ END
    #
    # 我们只声明了两个 hook、没有写任何 StateGraph 代码，框架自动把它们编译成图节点并
    # 接进了 ReAct 循环。想亲眼确认这张图，可以看 main() 里打印的 draw_mermaid() 结果。
    return create_agent(
        model=advanced_model,
        # get_weather 用于触发工具循环；另外两个依赖 Store，保留是为了让这个 Agent 的
        # 骨架与 @dynamic_prompt 实验一致。
        tools=[get_weather, get_user_info, save_user_info],
        middleware=[count_model_calls, inspect_model_output, observe_model_call, local_cache_shortcut, provider_call_probe],
        system_prompt="You are a helpful assistant",
        context_schema=Context,
        # 在这里声明扩展后的 State。这一行和装饰器上的声明若都去掉，hook 返回的
        # model_call_count / last_tool_names 会被**静默丢弃**（见 CallCountState 注释）。
        state_schema=CallCountState,
        store=store,
    )
    

# --- 六、辅助函数 -----------------------------------------------------------
def print_messages(result: dict) -> None:
    """按时间顺序打印这次运行产生的全部消息，便于观察 ReAct 循环轨迹。"""
    for message in result["messages"]:
        message.pretty_print()


def print_state_summary(result: dict) -> None:
    """
    打印 hook 写进 State 的字段 —— 本实验**最关键的验收点**。

    这两个值有内容，就证明"before_model / after_model 的返回值真的被当作 State 更新
    合并了"，而不只是打印了一下。对比 @dynamic_prompt 实验：那里无论怎么改提示词，
    最终 State 里都找不到任何痕迹。
    """
    print("=== 最终 State 摘要 ===")
    print(f"model_call_count = {result.get('model_call_count')}  （模型被调用的次数）")
    print(f"last_tool_names  = {result.get('last_tool_names')}  （最后一次模型请求的工具）")


# --- 七、运行入口 -----------------------------------------------------------
def main() -> None:
    """
    用真实模型演示 hook 行为。

    ⚠️ 这是**手动实验**，会产生外部 LLM API 调用与费用；运行前需确认 .env 与网络可用。
       仓库当前没有测试框架，所以这里的结果属于手动验证，不是自动化测试。
    """
    # .env 已在模块顶部加载（见第一节），这里不再重复调用。

    # 同一个程序运行期间只创建一份 Store 和 Agent。
    store = build_store()
    agent = build_agent(store)

    # 先打印静态图结构。这是零成本的，且能直观看到两个 hook 被编译成了图节点。
    # 注意：draw_mermaid() 只是静态图描述，不代表真实执行轨迹。
    print("=== create_agent 生成的图结构（静态） ===")
    print(agent.get_graph().draw_mermaid())

    # ---- 请求 1：不触发工具，期望两个 hook 各只触发 1 次 ----

    # 清零，避免受到前面请求统计结果的影响。
    provider_call_count["value"] = 0

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

    # 重新清零，只观察本次请求。
    provider_call_count["value"] = 0

    print("\n=== 请求 2：问天气（期望触发工具循环） ===")

    result02 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "你好呀，你知道我是谁吗？",
                },
                {
                    "role": "assistant",
                    "content": "你是？",
                },
                {
                    "role": "user",
                    "content": "我是猪八戒。",
                },
                {
                    "role": "assistant",
                    "content": "有什么我可以帮你的吗？",
                },
                {
                    "role": "user",
                    "content": "请告诉我北京今天的天气，并本地缓存下来。",
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
