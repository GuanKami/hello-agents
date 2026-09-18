"""
实验名称：工具权限校验与执行短路实验

实验目标：
    理解工具权限检查应该发生在真实工具执行之前，并使用 wrap_tool_call
    实现一个最小的工具权限闸门。

解决的问题：
    区分“模型已经提出工具调用”和“应用真正允许工具执行”两个阶段，
    验证应用代码可以根据可信的 Runtime Context 拒绝模型提出的工具调用，
    同时把拒绝原因作为 ToolMessage 交回 Agent，而不是让整个 Agent 直接崩溃。

使用的 Agent 概念：
    create_agent、ReAct 工具循环、ToolCallRequest、ToolMessage、
    wrap_tool_call、Runtime Context、handler 短路和工具权限控制。

系统架构：
    用户输入 -> 模型生成 AIMessage(tool_calls) -> weather_permission_guard
    -> authority=admin：调用 handler 执行 get_weather
    -> authority=user：不调用 handler，直接返回权限错误 ToolMessage
    -> 下一轮模型读取工具结果并生成最终回答。

实现方式：
    Context 由应用在 invoke 时注入，携带当前用户的 user_id 和 authority；
    weather_permission_guard 从 request.runtime.context 读取 authority，
    而不是相信模型放在 tool_call 参数中的权限信息。允许路径调用
    handler(request)，拒绝路径直接返回保留原 tool_call_id 的 ToolMessage。

验证方式：
    使用完全相同的天气问题执行两次，只改变 authority：管理员路径应真正
    执行 get_weather，普通用户路径应跳过 handler 和真实工具，并由 Agent
    根据权限错误 ToolMessage 生成最终回答。

学习总结：
    wrap_tool_call 不是独立的 StateGraph 节点，而是工具执行链中的包裹器。
    不调用 handler(request) 就可以在工具执行前短路；返回一个正确关联
    tool_call_id 的 ToolMessage 后，Agent 仍然可以继续下一轮模型处理。

已知限制：
    使用真实模型运行会产生 API 成本；get_weather 只是固定文本的演示工具；
    当前只演示单个天气工具和 admin/user 两种权限，没有实现通用权限矩阵、
    审计日志、异常转换、重试、超时或 Human-in-the-loop 审批。

后续优化方向：
    将单个工具判断扩展成可配置的工具权限策略，再单独增加工具异常转换、
    有限重试和高风险工具人工审批实验。
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

from tools import Context, get_weather


load_dotenv()

# 创建模型对象只完成客户端配置，不会立即发起模型请求。
# 真实 API 请求发生在 agent.invoke() 时。
advanced_model = ChatOpenAI(
    model=os.getenv("ADVANCED_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


@wrap_tool_call
def weather_permission_guard(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """在 get_weather 真正执行前做权限校验。

    request.tool_call 是模型生成的结构化调用意图，包含工具名、参数和
    tool_call_id；request.runtime.context 则是应用注入的可信运行时信息。
    权限判断使用后者，避免模型通过工具参数伪造 authority。

    允许路径：
        调用 handler(request)，继续下游工具执行链，最终执行真实工具。

    拒绝路径：
        不调用 handler(request)，直接返回 ToolMessage。因此真实工具不会执行，
        但这条带有原始 tool_call_id 的 ToolMessage 仍会进入 Agent 消息 State，
        并被下一轮模型读取。
    """
    tool_call = request.tool_call
    tool_name = tool_call["name"]
    context = request.runtime.context
    authority = context.authority

    if tool_name == "get_weather" and authority != "admin":
        print(
            "[tool_guard] 拒绝工具调用："
            f"tool={tool_name}, user_id={context.user_id}, authority={authority}"
        )

        # 这里故意不调用 handler(request)。这就是工具执行前的短路：
        # 下游 wrapper 和真实 get_weather 都不会被执行。
        return ToolMessage(
            content="工具调用被权限策略拒绝：当前用户无权调用天气工具。",
            name=tool_name,
            tool_call_id=tool_call["id"],
        )

    print(
        "[tool_guard] 允许工具调用："
        f"tool={tool_name}, user_id={context.user_id}, authority={authority}"
    )

    # 允许路径才继续执行下游 handler；handler 最终会抵达真实工具执行器。
    result = handler(request)
    print(f"[tool_guard] 下游 handler 执行完成，返回类型 = {type(result).__name__}")
    return result


def build_agent():
    """创建只包含权限 guard 的最小 Agent。

    这里不注册 observe_tool_call，是为了让本实验只聚焦一个问题：
    权限 middleware 是否调用 handler。管理员路径会打印 handler 完成日志，
    普通用户路径不会出现该日志，从而形成清晰的 1 次 / 0 次对照。
    """
    return create_agent(
        model=advanced_model,
        tools=[get_weather],
        middleware=[weather_permission_guard],
        system_prompt=(
            "You are a helpful assistant. "
            "When the user asks about weather, always use the get_weather tool."
        ),
        context_schema=Context,
    )


def print_messages(result: dict) -> None:
    """按时间顺序打印消息，观察权限错误如何回到 ReAct 循环。"""
    for message in result["messages"]:
        message.pretty_print()


def main() -> None:
    agent = build_agent()
    weather_question = "请告诉我北京今天的天气。"

    print("=== create_agent 生成的图结构（静态） ===")
    print(agent.get_graph().draw_mermaid())

    # 两次请求使用完全相同的用户输入，只改变 Context.authority，
    # 这样实验变量只有权限，便于比较 handler 是否被调用。
    print("\n=== 请求 1：管理员查询天气（期望工具正常执行） ===")
    result01 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user", 
                    "content": weather_question
                }
            ]
        },
        context=Context(
            user_id="user_1", 
            authority="admin"
        ),
    )
    print_messages(result01)

    print("\n========= 分割线 =========")

    print("\n=== 请求 2：普通用户查询天气（期望工具被权限短路） ===")
    result02 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user", 
                    "content": weather_question
                }
            ]
        },
        context=Context(
            user_id="user_2", 
            authority="user"
        ),
    )
    print_messages(result02)


if __name__ == "__main__":
    main()
