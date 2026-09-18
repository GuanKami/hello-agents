"""
实验名称：wrap_tool_call 工具异常转换实验

实验目标：
    理解真实工具执行过程中发生 Python 异常时，wrap_tool_call 如何介入工具执行链，
    并把异常转换成 Agent 可以继续处理的 ToolMessage。

解决的问题：
    区分“模型没有调用工具”和“工具已经被调用但执行失败”两种情况，验证工具异常
    不会直接导致 Agent 崩溃，而是可以作为工具结果回到下一轮模型上下文。

使用的 Agent 概念：
    create_agent、ReAct 工具循环、ToolCallRequest、wrap_tool_call、handler、
    ToolMessage、tool_call_id 和 Runtime Context。

系统架构：
    用户输入 -> 模型生成 divide tool_call -> wrap_tool_call 调用 handler
    -> divide 工具成功返回或抛出异常 -> ToolMessage -> 下一轮模型生成回答。

实现方式：
    handle_tool_error 包装每次工具执行。正常情况下原样返回下游 handler 的结果；
    捕获 ZeroDivisionError 时，不把 Python traceback 直接暴露给模型，而是构造一条
    带原始 tool_call_id 的可读错误 ToolMessage。

验证方式：
    请求 1 计算 10 / 2，验证正常工具调用链；请求 2 计算 10 / 0，验证模型确实
    调用 divide、工具抛出 ZeroDivisionError、middleware 捕获异常以及 Agent 继续
    完成下一轮模型调用。

学习总结：
    注册工具不代表模型一定调用工具；只有出现 AIMessage.tool_calls 后，工具 wrapper
    才会进入执行链。handler(request) 才是继续下游并执行真实工具的边界。

已知限制：
    当前只转换 ZeroDivisionError；其他异常会继续向上抛出。实验使用真实模型会产生
    API 成本，divide 是演示工具，错误消息也是固定的教学文本。

后续优化方向：
    增加 ValueError、TimeoutError、ConnectionError 等明确异常的分类转换，再研究限定
    异常类型、重试次数和退避策略的工具重试，最后接入超时控制和 Human-in-the-loop。
"""

import os
from collections.abc import Callable

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from tools import Context, divide

load_dotenv()

# 创建模型对象只完成客户端配置，不会立即发起模型请求。
# 真实 API 请求发生在 agent.invoke() 时。
advanced_model = ChatOpenAI(
    model=os.getenv("ADVANCED_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

@wrap_tool_call
def handle_tool_error(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """把指定工具异常转换为 ToolMessage，保持 ReAct 循环可以继续。

    request.tool_call 来自模型生成的结构化工具调用意图；它包含工具名称、参数和
    tool_call_id。此时工具还没有执行，真正执行发生在 handler(request) 内部。

    这里只捕获 ZeroDivisionError，是为了明确展示“已知、可恢复的业务异常”。其他
    未被处理的异常会继续向上抛出，避免 middleware 把未知编程错误静默吞掉。
    """
    tool_call = request.tool_call

    try:
        # handler 表示继续下游工具执行链；正常情况下它最终会调用 divide 工具，
        # 并返回 ToolMessage 或 Command。
        return handler(request)
    except ZeroDivisionError as exc:
        # 这里只记录异常类型，不把 traceback 或内部实现细节直接交给模型。
        print(f"[tool_error] 捕获工具异常：{type(exc).__name__}")

        # ToolMessage 是工具结果消息。保留原 tool_call_id，才能让 Agent 将错误结果
        # 与刚才的 divide 调用对应起来，并把它交给下一轮模型。
        return ToolMessage(
            content="工具执行失败：除数不能为 0。",
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        )


def build_agent():
    """创建本实验的 Agent，并明确要求除法请求必须经过工具。

    工具调用由模型决定，tools=[divide] 只表示模型拥有这个工具，并不保证每次都会
    调用它。这里的系统提示词是实验控制条件：如果不强制模型调用 divide，模型可能
    直接回答“除以零未定义”，导致 divide 和 ZeroDivisionError 都没有发生。
    """
    return create_agent(
        model=advanced_model,
        tools=[divide],
        middleware=[handle_tool_error],
        system_prompt=(
            "You are a tool-using calculator. "
            "For every division request, you must call the divide tool. "
            "Never answer a division request directly. "
            "Even when the divisor is zero, you must still call divide "
            "so the tool can report the execution error."
        ),
        context_schema=Context,
    )


def print_messages(result: dict) -> None:
    """按 State 中的顺序打印消息，观察工具异常如何回到 ReAct 循环。"""
    for message in result["messages"]:
        message.pretty_print()


def main() -> None:
    agent = build_agent()

    # draw_mermaid() 只生成编译后图的静态描述，不会调用模型或执行工具。
    print("=== create_agent 生成的图结构（静态） ===")
    print(agent.get_graph().draw_mermaid())

    # invoke() 才会真正启动 Agent；每次 invoke 都会经历模型决策和可能的工具循环。
    print("\n=== 请求 1：正常工具路径 ===")
    result01 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请计算 10 除以 2。",
                }
            ]
        },
        # 本实验不使用权限判断，但 Context schema 要求 invoke 注入完整 Context。
        # user_id / authority 在本实验中只是运行时上下文示例，不参与 divide 计算。
        context=Context(
            user_id="user_1",
            authority="user",
        ),
    )
    print_messages(result01)

    print("\n========= 分割线 =========")

    # 这里预期模型仍然必须调用 divide；ZeroDivisionError 会在 handler 内部发生，
    # 然后由 handle_tool_error 转换成 ToolMessage，而不是让 invoke 直接失败。
    print("\n=== 请求 2：工具异常路径 ===")
    result02 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请计算 10 除以 0。",
                }
            ]
        },
        context=Context(
            user_id="user_2",
            authority="user",
        ),
    )
    print_messages(result02)


if __name__ == "__main__":
    main()
