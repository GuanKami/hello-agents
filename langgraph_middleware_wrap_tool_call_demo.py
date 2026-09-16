"""
实验名称：wrap_tool_call 工具调用观察实验

实验目标：
    观察 create_agent 内部的工具执行链，理解 @wrap_tool_call 在工具真正执行前后
    如何拦截一次具体的工具调用。

解决的问题：
    区分“模型生成 tool_calls”与“框架实际执行工具函数”这两个阶段，并确认工具调用
    的名称、参数、tool_call_id 以及 ToolMessage 结果如何在链路中传递。

使用的 Agent 概念：
    Tool Calling、ReAct 工具循环、create_agent、wrap_tool_call、ToolCallRequest、
    ToolMessage、Runtime Context 和 Store。

系统架构：
    用户消息 -> 模型节点 -> AIMessage(tool_calls) -> tools 节点 ->
    observe_tool_call -> 真实 @tool 函数 -> ToolMessage -> 模型节点。
    observe_model_call 作为辅助观察层，记录每轮模型响应；observe_tool_call 位于工具
    执行链内部，不是独立的 StateGraph 节点。

实现方式：
    使用 create_agent 注册 get_weather、get_user_info 和 save_user_info，通过
    @wrap_tool_call 包装每次工具执行，在 handler(request) 前后打印调用参数和结果。
    使用 InMemoryStore 预置用户资料，用 Context 注入当前 user_id 和 authority。

验证方式：
    请求 1 使用普通寒暄，验证没有 tool_calls 时 observe_tool_call 不触发；请求 2
    触发天气和用户资料工具，验证每次具体工具调用都会分别进入 wrapper，且工具调用
    ID 与返回 ToolMessage 的 tool_call_id 一致。

学习总结：
    LLM 先提出工具调用意图，LangGraph 再进入工具执行链。handler(request) 才会
    真正执行工具函数；wrapper 可以在执行前观察或拦截，在执行后观察 ToolMessage，
    然后把结果交回 ReAct 循环供下一轮模型读取。

已知限制：
    本实验使用真实模型，结果受模型决策影响并会产生 API 调用成本；get_weather 是
    固定返回文本的演示工具；InMemoryStore 只在当前 Python 进程内有效；当前只做
    工具调用观察，没有实现工具短路、异常转换、重试或人工审批。

后续优化方向：
    增加不调用 handler 的工具短路实验，再实现工具异常到 ToolMessage 的转换、有限
    重试、权限校验和 Human-in-the-loop 审批。
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

@wrap_tool_call
def observe_tool_call(
    request: ToolCallRequest, 
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """
    观察一次工具调用的完整过程。

    handler(request) 之前：
        模型已经生成 tool_calls；
        但具体工具函数还没有真正执行。

    handler(request)：
        执行本次工具调用。

    handler(request) 之后：
        得到 ToolMessage 或 Command；
        该结果随后会进入 Agent State，并被下一轮模型读取。
    """

    tool_call = request.tool_call

    print("\n[wrap_tool] 准备执行工具")
    print(f"[wrap_tool] tool_name = {tool_call['name']}")
    print(f"[wrap_tool] tool_args = {tool_call['args']}")
    print(f"[wrap_tool] tool_call_id = {tool_call['id']}")

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
    return create_agent(
        model=advanced_model,
        tools=[get_weather, get_user_info, save_user_info],
        middleware=[observe_model_call, observe_tool_call],
        system_prompt="You are a helpful assistant",
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

    # ---- 请求 1：不触发工具 ----

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

    print("\n========= 分割线 =========")

    # ---- 请求 2：触发工具 ----
    # 这次调用也验证了：没有 checkpointer 时，上一次的计数不会带过来，

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

if __name__ == "__main__":
    main()
