"""
实验名称：wrap_tool_call 工具调用观察实验

实验目标：
    理解模型生成工具调用意图之后，LangGraph Agent 如何进入工具执行链，
    以及 wrap_tool_call 如何在真实工具执行前后观察这条链路。

解决的问题：
    区分“模型请求调用工具”和“工具函数已经真正执行”这两个不同阶段，
    观察工具名称、参数、tool_call_id 与 ToolMessage 如何在 ReAct 循环中传递。

使用的 Agent 概念：
    create_agent、ReAct 工具循环、ToolCallRequest、ToolMessage、
    wrap_model_call、wrap_tool_call、Runtime Context 和 Store。

系统架构：
    用户输入 -> 模型生成 AIMessage(tool_calls) -> observe_tool_call
    -> create_agent 内部的工具执行器 -> 真实 @tool 函数
    -> ToolMessage -> 下一轮模型 -> 最终回答。

实现方式：
    observe_model_call 记录每轮模型输入和输出；observe_tool_call 接收
    ToolCallRequest，在 handler(request) 前后记录工具执行信息；build_store
    通过 InMemoryStore 为用户资料工具提供精确读取和写入的运行时依赖。
    本文件不负责权限控制，权限实验拆分到 langgraph_middleware_tool_guard_demo.py。

验证方式：
    请求 1 使用普通寒暄，验证没有 tool_calls 时工具 wrapper 不触发；请求 2
    同时询问用户资料和天气，验证每个实际工具调用都会独立进入 observe_tool_call，
    并在执行后把 ToolMessage 交给下一轮模型。

学习总结：
    wrap_tool_call 不是独立的 StateGraph 节点，而是工具执行链中的包裹器。
    handler(request) 表示继续下游工具执行链；在本实验中，继续后最终会执行
    真实工具函数。工具 wrapper 的观察点发生在工具执行链内部，而不是模型节点外部。

已知限制：
    使用真实模型运行会产生 API 成本，工具是否被调用受模型决策影响；
    get_weather 只是固定文本的演示工具；当前只验证工具调用观察，还没有实现
    工具异常转换、重试、超时、权限控制和 Human-in-the-loop 审批。
    InMemoryStore 只在当前 Python 进程内有效。

后续优化方向：
    保持本文件作为工具观察基线；工具权限控制见
    langgraph_middleware_tool_guard_demo.py，后续再单独增加工具异常转换与有限重试实验。
"""

import os
from collections.abc import Callable

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from tools import Context, get_weather, get_user_info, save_user_info


load_dotenv()

# 本实验使用高能力模型，便于它根据问题选择用户资料和天气工具。
# 创建 ChatOpenAI 对象本身不会执行模型请求；真正的请求发生在 agent.invoke() 中。
advanced_model = ChatOpenAI(
    model=os.getenv("ADVANCED_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


def build_store() -> InMemoryStore:
    """创建并预置本次进程使用的长期资料 Store。

    get_user_info 和 save_user_info 会通过 runtime.store 访问这里的用户资料；
    get_weather 不读取 Store。将 Store 作为依赖注入，是为了让工具不需要知道
    具体使用 InMemoryStore 还是未来的 SqliteStore。

    InMemoryStore 只在当前 Python 进程中保存数据，进程退出后资料会丢失。
    """
    store = InMemoryStore()

    # namespace ("users",) + key "user_1" 构成结构化资料的精确寻址路径。
    # 姓名、语言等是事实字段，应使用精确 key 读取，而不是使用语义 search() 猜测。
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

    messages = request.state.get("messages", [])

    # 这里观察的是模型调用前的工作流消息数量，不代表已经发生了网络请求。
    print(f"[wrap] 消息数 = {len(messages)}")
    print(f"[wrap] system_message = {request.system_message}")

    # 继续下游模型 wrapper。若某个更内层 wrapper 直接返回 ModelResponse，
    # 这里仍然能拿到响应，但不能把 handler 简单等同于“必然访问 provider”。
    response = handler(request)

    # response.result[0] 是本轮模型产生的 AIMessage。
    # tool_calls 表示模型提出了工具调用意图，尚不能证明工具已经执行。
    ai_message = response.result[0]
    print(
        "[wrap] 下游响应的 tool_calls = "
        f"{[tool_call['name'] for tool_call in (ai_message.tool_calls or [])]}"
    )

    # 真实 provider 响应通常带有元数据和 token 使用量；本地短路构造的
    # AIMessage 往往没有这些信息，因此它们可作为观察“是否经过真实模型”的辅助证据。
    print(f"[wrap] response_metadata = {ai_message.response_metadata}")
    print(f"[wrap] usage_metadata = {ai_message.usage_metadata}")
    print(f"[wrap] message_id = {ai_message.id}")

    return response


@wrap_tool_call
def observe_tool_call(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """观察一次工具调用从请求到结果的完整过程。

    request.tool_call 是模型已经生成的结构化调用意图，其中包含工具名、参数
    和 tool_call_id。此时只能说明模型想调用工具，具体工具函数还没有执行。

    handler(request) 表示继续工具执行链。对于没有被其他 wrapper 拦截的请求，
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

    # 只有继续调用 handler，工具执行链才会向真实工具推进。
    result = handler(request)

    print(f"[wrap_tool] 工具执行完成，返回类型 = {type(result).__name__}")

    if isinstance(result, ToolMessage):
        print(f"[wrap_tool] tool_message.name = {result.name}")
        print(f"[wrap_tool] tool_message.content = {result.content}")
        print(f"[wrap_tool] tool_message.tool_call_id = {result.tool_call_id}")
    else:
        print("[wrap_tool] 工具返回了 Command")

    return result


def build_agent(store: InMemoryStore):
    """组装模型观察器、工具观察器和用户资料工具。

    本实验的重点是观察工具调用，不注册 weather_permission_guard。
    这样可以把“工具是否被调用”和“工具是否有权限调用”拆成两个独立实验。
    当前工具 wrapper 的执行链只有：

        observe_tool_call
            -> create_agent 内部的真实工具执行器
    """
    return create_agent(
        model=advanced_model,
        tools=[get_weather, get_user_info, save_user_info],
        middleware=[observe_model_call, observe_tool_call],
        system_prompt=(
            "You are a helpful assistant. "
            "When the user asks about weather, use get_weather. "
            "When the user asks about their profile or identity, use get_user_info. "
            "When the user asks to save profile information, use save_user_info."
        ),
        context_schema=Context,
        store=store,
    )


def print_messages(result: dict) -> None:
    """按时间顺序打印本次运行产生的全部消息，观察 ReAct 循环轨迹。"""
    for message in result["messages"]:
        message.pretty_print()


def main() -> None:
    store = build_store()
    agent = build_agent(store)

    print("=== create_agent 生成的图结构（静态） ===")
    print(agent.get_graph().draw_mermaid())

    # 请求 1：普通寒暄。
    # 没有工具需求时，模型只返回普通 AIMessage，observe_tool_call 不会触发。
    print("\n=== 请求 1：寒暄（期望不触发工具 wrapper） ===")
    result01 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好"}]},
        context=Context(user_id="user_1", authority="user"),
    )
    print_messages(result01)

    print("\n========= 分割线 =========")

    # 请求 2：一次问题同时包含结构化资料查询和天气查询。
    # 模型可能按顺序或在同一轮提出多个 tool_call；每个实际执行的工具都会
    # 分别进入 observe_tool_call，并将各自的 ToolMessage 交回下一轮模型。
    print("\n=== 请求 2：查询资料和天气（期望触发多个工具调用） ===")
    result02 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请查询我的个人资料，并告诉我北京今天的天气。",
                }
            ]
        },
        context=Context(user_id="user_1", authority="user"),
    )
    print_messages(result02)


if __name__ == "__main__":
    main()
