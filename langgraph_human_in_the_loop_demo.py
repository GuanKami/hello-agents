"""
实验名称：Human-in-the-loop 工具审批实验

实验目标：
    学习 Agent 在执行需要人工确认的工具前如何暂停，
    并在人工决定后通过同一个 thread_id 恢复执行。

解决的问题：
    防止模型未经人工确认直接执行高风险工具。

使用的 Agent 概念：
    HumanInTheLoopMiddleware、interrupt、checkpoint、
    thread_id、Command(resume=...)、Tool Calling。

系统架构：
    用户请求 -> 模型生成工具调用
    -> HITL middleware 拦截
    -> interrupt 暂停
    -> 人工审批
    -> Command(resume=...)
    -> 执行工具或拒绝工具
    -> 模型生成最终回答。

实现方式：
    使用 HumanInTheLoopMiddleware 对 divide 工具配置审批，
    使用 InMemorySaver 保存当前执行状态。

验证方式：
    第一次 invoke 触发审批中断，返回 __interrupt__；
    恢复时使用相同 thread_id 和 Command(resume=...) 提交人工决定。
    approve 路径已验证：工具执行并返回正常 ToolMessage；
    reject 路径已验证：工具不执行，返回拒绝 ToolMessage，模型不再次重试。

学习总结：
    interrupt 会暂停当前 Agent 执行，checkpointer 保存恢复所需的 State，
    thread_id 用于定位这次暂停的执行上下文。恢复时再次看到原来的
    AIMessage/tool_call 是正常现象，因为 Agent 正在从“待审批的工具调用”继续；
    reject 只拒绝当前工具动作，不等于无条件终止整个任务，是否重试还受模型提示词影响。

已知限制：
    InMemorySaver 只在当前 Python 进程有效，进程退出后数据丢失；
    当前 main 主要演示 reject，edit/respond 决策和跨进程恢复尚未验证；
    模型是否在拒绝后重试受系统提示词和模型行为影响。

后续优化方向：
    切换 SqliteSaver，验证进程重启后的恢复；
    验证 edit/respond 决策；增加审批页面、审批人身份、超时和审计日志。

                    ┌──────────────┐
                    │ 模型调用工具   │
                    └──────┬───────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ HumanInTheLoop   │
                 │ Middleware       │
                 └────────┬─────────┘
                          │
                          ▼
                    interrupt 暂停
                          │
                 保存 State / checkpoint
                          │
            ┌─────────────┴─────────────┐
            │                           │
        approve                       reject
            │                           │
            ▼                           ▼
       执行真实工具                 跳过工具
            │                           │
            ▼                           ▼
       ToolMessage                  拒绝结果
            │                           │
            └─────────────┬─────────────┘
                          ▼
                    下一轮模型
"""

import os

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tools import Context, divide

load_dotenv()

# 这里使用真实模型，是为了观察“模型生成 tool_call -> 中断 -> 恢复”的完整链路。
# 运行脚本会产生模型调用费用；模型是否生成 divide tool_call 仍受提示词和模型行为影响。
advanced_model = ChatOpenAI(
    model=os.getenv("ADVANCED_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        # True 表示 divide 的执行前需要人工审批。middleware 会先暂停 Agent，
        # 再把 action_requests 和 review_configs 放进 __interrupt__，等待外部决定。
        "divide": True,
    },
    # 这个前缀会出现在中断描述中，帮助审批方理解为什么需要确认。
    description_prefix="工具执行需要人工审批",
)

# checkpoint 保存的是当前 Agent 的 State 和待恢复位置，不是用户长期资料。
# 这里使用 InMemorySaver 只为学习暂停/恢复机制；它不能跨 Python 进程保存。
checkpointer = InMemorySaver()

def build_agent():
    """构造带人工审批中间件和 checkpoint 的计算 Agent。

    执行顺序是：模型生成 divide tool_call -> middleware interrupt -> 人工决定
    -> Command(resume=...) 恢复 -> approve 执行工具，或 reject 生成拒绝 ToolMessage。
    """
    return create_agent(
        model=advanced_model,
        tools=[divide],
        middleware=[hitl_middleware],
        checkpointer=checkpointer,
        context_schema=Context,
        system_prompt=(
            "You are a tool-using calculator. "
            "For each new division request, call the divide tool once. "
            "If a tool result says that the user rejected the tool call, "
            "do not call divide again and do not retry with different arguments. "
            "Explain that the operation was not approved and end the task."
        ),
    )

def print_messages(result: dict) -> None:
    """按时间顺序打印本次结果中的消息，观察中断前后的 ReAct 轨迹。

    恢复调用可能再次显示原来的 AIMessage/tool_call，这是从 checkpoint 恢复
    待审批动作的表现，不代表工具已经执行了；是否执行要看后面的 ToolMessage。
    """
    for message in result["messages"]:
        message.pretty_print()

def main() -> None:
    agent = build_agent()

    # 两次 invoke 必须使用同一个 thread_id，才能找到第一次中断时保存的 checkpoint。
    # 换 thread_id 等于开启另一条执行链，无法恢复当前待审批动作。
    thread_id = "hitl-reject-20260918"
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    runtime_context = Context(
        user_id="user_1",
        authority="user",
    )

    # 第一次调用只负责让模型提出工具调用。由于 divide 配置了 HITL，
    # Agent 会在真正执行 divide 之前暂停，并把中断信息放到 __interrupt__。
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请计算 10 除以 2。",
                }
            ]
        },
        config=config,
        context=runtime_context,
    )

    print_messages(result)

    interrupts = result.get("__interrupt__")
    print("中断信息：")
    print(interrupts)

    if not interrupts:
        print("没有触发中断，请检查模型是否生成了 divide tool_call。")
        return

    # 这里模拟人工拒绝。Command(resume=...) 不会重新提交用户问题，
    # 而是告诉被 checkpoint 暂停的 Agent 如何处理当前待审批动作。
    resumed_result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "reject",
                    }
                ]
            }
        ),
        config=config,
        context=runtime_context,
    )

    print_messages(resumed_result)

    # 当前系统提示词要求拒绝后不要重试，因此正常结果不会再次产生中断。
    # 如果这里仍有 __interrupt__，说明模型又提出了新的工具调用，需要继续审批。
    resumed_interrupts = resumed_result.get("__interrupt__")
    if resumed_interrupts:
        print("拒绝后仍产生新的中断：模型又提出了新的工具调用。")
        print(resumed_interrupts)
    else:
        print("拒绝后没有新的中断，Agent 已结束本次任务。")

if __name__ == "__main__":
    main()
