"""
实验名称：wrap_tool_call 工具观察与权限短路实验

实验目标：
    理解模型生成工具调用意图之后，LangGraph Agent 如何进入工具执行链，
    以及 wrap_tool_call 如何在真实工具执行前后观察或改变这条链路。

解决的问题：
    区分“模型请求调用工具”和“工具函数已经真正执行”这两个不同阶段，
    并验证权限中间件可以在不执行真实工具的情况下返回一个 ToolMessage，
    让 Agent 继续完成后续模型推理。

使用的 Agent 概念：
    create_agent、ReAct 工具循环、ToolCallRequest、ToolMessage、
    wrap_model_call、wrap_tool_call、Runtime Context 和工具权限控制。

系统架构：
    用户输入 -> 模型生成 tool_calls -> wrap_tool_call 权限检查
    -> 允许时执行 get_weather -> ToolMessage -> 模型生成最终回答
    -> 拒绝时直接返回权限错误 ToolMessage -> 模型生成解释性回答。

实现方式：
    observe_model_call 观察模型响应；weather_permission_guard 位于工具
    wrapper 链的外层，使用 request.runtime.context.authority 做权限判断；
    observe_tool_call 记录真正进入 handler 的工具调用前后信息。

验证方式：
    请求 1 使用 authority="admin"，验证工具被允许执行；请求 2 使用
    authority="user"，验证权限中间件不调用下游 handler，真实工具和内层
    observe_tool_call 都不会执行，但 Agent 仍能读取拒绝信息并返回最终回答。

学习总结：
    wrap_tool_call 不是独立的 StateGraph 节点，而是工具执行链中的包裹器。
    handler(request) 表示继续下游 wrapper；只有链路最终继续到底层工具时，
    工具函数才会真正运行。

已知限制：
    使用真实模型运行会产生 API 成本；get_weather 只是固定文本的演示工具；
    当前只验证了观察和权限短路，还没有实现异常转换、有限重试、超时和审批。
    InMemoryStore 仅作为 Agent 的依赖注入示例，当前 get_weather 不读取 Store。

后续优化方向：
    增加工具异常到 ToolMessage 的统一转换，再实现限定异常类型、次数和退避
    策略的工具重试，最后进入 Human-in-the-loop 审批和可恢复执行实验。
"""

import os
from collections.abc import Callable

from dotenv import load_dotenv

from langchain.agents.middleware import wrap_tool_call, wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from tools import Context, get_weather, get_user_info, save_user_info
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent


load_dotenv()

# 高费率模型
advanced_model = ChatOpenAI(
    model=os.getenv("ADVANCED_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

def build_store() -> InMemoryStore:
    """
    创建本次程序运行期间共享的长期记忆 Store。

    做成函数返回值（依赖注入）而不是模块级全局单例，是为了让"运行入口"决定用哪个后端；
    Agent 和工具只通过 runtime.store 访问它，不关心数据存在哪里。换 SqliteStore 只改这里。

    InMemoryStore 的数据只活在当前 Python 进程内，进程退出即全部丢失。
    当前实验的 get_weather 不读取 Store；这里保留 Store 是为了演示 create_agent 的
    依赖注入位置，并为后续把权限、用户资料或工具结果接入长期记忆留出扩展点。
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

@wrap_model_call
def observe_model_call(request: ModelRequest, handler) -> ModelResponse:
    """观察模型 wrapper 的输入和输出，但不改变模型请求。

    这个 wrapper 位于模型节点内部。request.state 可以读取当前工作流 State，
    但本函数没有把观察数据写回 State。handler(request) 的含义是“继续进入
    下一个模型 wrapper”；它可能还会经过其他 wrapper，只有下游链最终没有
    被短路时才会真正请求模型服务商。
    """

    # 这里观察的是模型调用前的工作流消息数量，不代表已经发生了网络请求。
    print(f"[wrap] 消息数 = {len(request.state['messages'])}")
    print(f"[wrap] system_message = {request.system_message}")

    # 继续下游模型 wrapper。若某个更内层 wrapper 直接返回 ModelResponse，
    # 这里仍然能拿到响应，但不应把 handler 简单等同于“必然访问 provider”。
    response = handler(request)

    # response.result[0] 是本轮模型产生的 AIMessage。
    # tool_calls 表示模型提出了工具调用意图，尚不能证明工具已经执行。
    ai_msg = response.result[0]
    print(f"[wrap] 下游响应的 tool_calls = {[tc['name'] for tc in (ai_msg.tool_calls or [])]}")

    # 真实 provider 响应通常带有元数据和 token 使用量；本地短路构造的
    # AIMessage 往往没有这些信息，因此它们可作为观察“是否经过真实模型”的辅助证据。
    print(f"[wrap] response_metadata = {ai_msg.response_metadata}")
    print(f"[wrap] usage_metadata = {ai_msg.usage_metadata}")
    print(f"[wrap] message_id = {ai_msg.id}")

    return response

@wrap_tool_call
def observe_tool_call(
    request: ToolCallRequest, 
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """
    观察一次工具调用从请求到结果的完整过程。

    request.tool_call 是模型已经生成的结构化调用意图，其中包含工具名、参数
    和 tool_call_id。此时只能说明模型想调用工具，具体工具函数还没有执行。

    handler(request) 表示继续工具执行链。对于没有被外层 wrapper 拦截的请求，
    它最终会执行对应的 Python 工具函数，并通常返回 ToolMessage；某些高级
    工具流程也可能返回 Command。

    handler(request) 之后得到的结果会回到 Agent 的消息 State，随后被下一轮
    模型读取。这个 wrapper 本身只观察，不改变工具请求或返回值。
    """

    tool_call = request.tool_call

    print("\n[wrap_tool] 准备执行工具")
    print(f"[wrap_tool] tool_name = {tool_call['name']}")
    print(f"[wrap_tool] tool_args = {tool_call['args']}")
    print(f"[wrap_tool] tool_call_id = {tool_call['id']}")

    # 只有执行到这里，才允许下游 wrapper 和真实工具继续运行。
    result = handler(request)

    print(f"[wrap_tool] 工具执行完成，返回类型 = {type(result).__name__}")

    if isinstance(result, ToolMessage):
        print(f"[wrap_tool] tool_message.name = {result.name}")
        print(f"[wrap_tool] tool_message.content = {result.content}")
        print(f"[wrap_tool] tool_message.tool_call_id = {result.tool_call_id}")
    else:
        print("[wrap_tool] 工具返回了 Command")

    return result

@wrap_tool_call
def weather_permission_guard(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    """
    在工具执行前对 get_weather 进行权限控制。

    admin：调用 handler(request)，继续进入下游 wrapper 和真实工具。

    user：不调用 handler(request)，直接返回一个与原 tool_call_id 对应的
    ToolMessage。因此真实的 get_weather 函数不会执行，内层 observe_tool_call
    也不会被触发，但 Agent 仍然可以把这条 ToolMessage 交给下一轮模型处理。
    """

    tool_call = request.tool_call
    tool_name = tool_call["name"]

    # Context 是应用代码在 invoke 时注入的可信运行时信息，
    # 不应该让模型通过 tool_call 参数自行声明或伪造权限。

    authority = request.runtime.context.authority

    if tool_name == "get_weather" and authority != "admin":
        print(
            "[tool_guard] 拒绝工具调用："
            f"tool={tool_name}, authority={authority}"
        )

        # 这里直接返回，没有调用 handler(request)。由于本函数在 middleware
        # 列表中位于 observe_tool_call 外层，拒绝路径会连内层观察器和真实
        # get_weather 一起跳过；保留原 tool_call_id 可让 Agent 正确关联结果。

        return ToolMessage(
            content="工具调用被权限策略拒绝：当前用户无权调用天气工具。",
            name=tool_name,
            tool_call_id=tool_call["id"],
        )

    print(
        "[tool_guard] 允许工具调用："
        f"tool={tool_name}, authority={authority}"
    )

    # 只有放行时才继续进入下游 wrapper；最终是否执行真实工具由下游链决定。
    return handler(request)


def build_agent(store: InMemoryStore):
    """组装模型观察器和工具执行器，并明确 wrapper 的嵌套顺序。

    middleware 列表中的工具 wrapper 形成如下调用链：

        weather_permission_guard
            -> observe_tool_call
                -> create_agent 内部的真实工具执行器

    因此权限拒绝时，guard 不调用 handler，内层 observe_tool_call 也不会出现
    日志；权限允许时，调用会依次穿过两个 wrapper 后执行 get_weather。
    """
    return create_agent(
        model=advanced_model,
        tools=[get_weather],
        middleware=[observe_model_call, weather_permission_guard, observe_tool_call],
        system_prompt=(
            "You are a helpful assistant. "
            "When the user asks about weather, use the get_weather tool."
            ),
        context_schema=Context,
        store=store,
    )

def print_messages(result: dict) -> None:
    """按时间顺序打印这次运行产生的全部消息，便于观察 ReAct 循环轨迹。"""
    for message in result["messages"]:
        message.pretty_print()

def main() -> None:
    store = build_store()
    agent = build_agent(store)

    print("=== create_agent 生成的图结构（静态） ===")
    print(agent.get_graph().draw_mermaid())

    # ---- 请求 1：管理员放行路径 ----
    # 两次请求使用相同的天气问题，只改变 Context.authority，便于观察
    # “权限允许”和“权限拒绝”对工具执行链的影响。
    print("\n=== 请求 1：管理员查询天气（期望工具正常执行） ===")
    result01 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "你好，请告诉我北京今天的天气",
                }
            ]
        },
        context=Context(
            user_id="user_1",
            authority="admin",
        )
    )
    print_messages(result01)

    print("\n========= 分割线 =========")

    # ---- 请求 2：普通用户拒绝路径 ----
    # 模型仍可能先生成 get_weather tool_call；权限中间件在工具真正执行前
    # 将其转换成 ToolMessage，所以这里验证的是“模型提出调用”与“工具执行”
    # 之间的安全边界，而不是模型是否会生成 tool_call。
    print("\n=== 请求 2：普通用户查询天气（期望工具被权限短路） ===")

    result02 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "你好，请告诉我北京今天的天气。",
                }
            ]
        },
        context=Context(
            user_id="user_2",
            authority="user",
        )
    )
    print_messages(result02)

if __name__ == "__main__":
    main()
