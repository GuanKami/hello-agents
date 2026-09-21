"""
实验名称：SqliteSaver 持久化 Human-in-the-loop 实验

实验目标：
    验证 Human-in-the-loop 在 Python 进程结束后，
    是否可以通过 SqliteSaver 和相同 thread_id 恢复。

解决的问题：
    InMemorySaver 只能在当前 Python 进程中保存 checkpoint。
    本实验验证跨 Python 进程恢复暂停的 Agent。

使用的 Agent 概念：
    HumanInTheLoopMiddleware
    interrupt
    SqliteSaver
    checkpoint
    thread_id
    Command(resume=...)
    Tool Calling

系统架构：
    进程 A：
        用户请求
        -> 模型生成 divide tool_call
        -> HumanInTheLoopMiddleware 中断
        -> SqliteSaver 写入 checkpoint
        -> 进程结束

    进程 B：
        重新创建 Agent
        -> 打开相同 SQLite 数据库
        -> 使用相同 thread_id
        -> Command(resume=...)
        -> 恢复工具执行或拒绝工具
        -> 模型生成最终回答

实现方式：
    使用 SqliteSaver 替代 InMemorySaver。
    使用 start 和 resume 两种命令模拟两个独立 Python 进程。

验证方式：
    1. 运行 start，触发中断并结束进程。
    2. 确认 SQLite 文件已经生成。
    3. 使用新的 Python 进程运行 resume。
    4. 使用相同 thread_id 恢复 approve 或 reject。
    5. 2026-09-21 已验证 test01 的 approve 跨进程恢复和 test02 的 reject
       跨进程恢复；不同 thread_id 无法恢复原任务的负向验证尚未单独执行。

学习总结：
    SqliteSaver 保存的是 Agent State、消息、待审批工具调用和恢复位置，
    不是用户长期资料。
    thread_id 是查询某次执行 checkpoint 的关键标识。
    Context 仍然属于本次运行时上下文，恢复时需要再次注入。
    start 和 resume 使用相同的工具调用 ID，说明 resume 是从 checkpoint
    恢复待审批动作，而不是重新提交用户消息并生成一个新的工具调用。

已知限制：
    当前只验证本地 SQLite 文件中的 approve 和 reject。
    edit、respond、审批身份、超时、审计、并发和数据库备份尚未实现。

后续优化方向：
    使用 PostgresSaver 等生产级 checkpoint。
    增加审批记录、审批用户、超时、幂等和不同 thread_id 的负向验证。
"""

from pathlib import Path
import os
import sys

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from tools import Context, divide

# start 和 resume 都需要重新加载模型配置，因为它们由两个独立的 Python 进程执行。
load_dotenv()

# 这里仍然使用真实模型，让模型生成需要审批的 divide tool_call。
# 运行 start 或 resume 都可能产生模型调用费用。
advanced_model = ChatOpenAI(
    model=os.getenv("ADVANCED_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

# 使用脚本所在目录构造绝对路径，避免 PowerShell 当前目录不同导致两次运行
# 打开不同的数据库文件。这个文件保存 checkpoint，不保存用户长期资料。
DB_PATH = Path(__file__).with_name("hitl-checkpoint.db")


def build_agent(checkpointer: SqliteSaver):
    """
    创建使用 SqliteSaver 的 HITL Agent。

    Agent 图结构必须在 start 和 resume 两个进程中保持一致，
    这样恢复时才能正确解释 checkpoint 中保存的执行位置。
    """

    # start 和 resume 必须构造相同的 Agent 图：工具、middleware 和系统提示词
    # 发生变化，都可能导致 checkpoint 中保存的执行位置无法按预期恢复。
    hitl_middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            # True 表示 divide 的执行前需要人工审批。middleware 会先暂停 Agent，
            # 再把 action_requests 和 review_configs 放进 __interrupt__，等待外部决定。
            "divide": True,
        },
        # 这个前缀会出现在中断描述中，帮助审批方理解为什么需要确认。
        description_prefix="工具执行需要人工审批",
    )

    return create_agent(
        model=advanced_model,
        tools=[divide],
        middleware=[hitl_middleware],
        checkpointer=checkpointer,
        context_schema=Context,
        # 拒绝后的 ToolMessage 会回到模型。这个提示词要求模型不要再次调用
        # divide，否则 reject 后可能又生成新的 tool_call 并触发新的中断。
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


def make_config(thread_id: str) -> dict:
    """
    根据 thread_id 创建 LangGraph checkpoint 配置。

    thread_id 是查找某次 Agent 执行记录的关键。start 和 resume 必须使用
    完全相同的值；换值就相当于查找另一条执行链。
    """
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def make_context() -> Context:
    """
    创建本次运行的运行时上下文。

    Context 是每次 invoke 注入的运行时信息，不等于 SqliteSaver 中保存的
    checkpoint。因此 resume 进程需要重新构造并传入 Context。
    """
    return Context(
        user_id="user_1",
        authority="user",
    )


def start_agent(thread_id: str) -> None:
    """
    在当前进程创建一次待审批的工具调用。

    进程结束后，checkpoint 仍然保存在 hitl-checkpoint.db 中，供另一个
    Python 进程恢复。这个函数只制造中断，不提交人工决定。
    """
    config = make_config(thread_id)
    context = make_context()

    # from_conn_string 会打开同一个 SQLite 文件；SqliteSaver 会在需要时
    # 创建 checkpoints 和 writes 等表。with 结束后连接关闭，但文件保留。
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        agent = build_agent(checkpointer)

        # 第一次 invoke 提交原始用户消息。模型生成 divide 后，HITL middleware
        # 在真正执行工具之前暂停，因此这里的返回值包含 __interrupt__。
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "请计算 2246 除以 10。",
                    }
                ]
            },
            config=config,
            context=context,
        )

        print("=== start 模式 ===")
        print(f"数据库文件：{DB_PATH.resolve()}")
        print(f"thread_id：{thread_id}")

        for message in result.get("messages", []):
            message.pretty_print()

        # __interrupt__ 不是普通对话消息，而是 LangGraph 暂停执行时返回的
        # 控制信息，里面包含待审批动作和允许的决定类型。
        interrupts = result.get("__interrupt__")

        print("=== 中断信息 ===")
        print(interrupts)

        if interrupts:
            print()
            print("Agent 已暂停。请结束当前进程，")
            print("然后在新的 Python 进程中运行 resume 模式。")
        else:
            print("没有产生中断，请检查模型是否生成了 divide tool_call。")


def resume_agent(thread_id: str, decision: str) -> None:
    """
    在新的 Python 进程中恢复之前暂停的 Agent。

    decision 可以是：
        approve
        reject

    resume 不重新提交用户消息，而是通过 thread_id 找到 SQLite 中保存的
    checkpoint，再用 Command 告诉 HITL middleware 如何处理待审批动作。
    """
    if decision not in {"approve", "reject"}:
        raise ValueError("decision 必须是 approve 或 reject")

    config = make_config(thread_id)
    context = make_context()

    # 这里重新打开持久化文件，并重新创建相同图结构的 Agent；这正是
    # InMemorySaver 做不到、而 SqliteSaver 可以做到的跨进程恢复。
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        agent = build_agent(checkpointer)

        # 这里故意不再传 {"messages": [...]}。原始消息、AIMessage、待审批
        # tool_call 和图的暂停位置，都应该从 SQLite checkpoint 中恢复。
        # 如果改用新的 thread_id，LangGraph 找不到对应的待审批执行。
        result = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {
                            "type": decision,
                        }
                    ]
                }
            ),
            config=config,
            context=context,
        )

        print("=== resume 模式 ===")
        print(f"数据库文件：{DB_PATH.resolve()}")
        print(f"thread_id：{thread_id}")
        print(f"人工决定：{decision}")

        for message in result.get("messages", []):
            message.pretty_print()

        # approve 正常会继续执行 divide；reject 会产生“工具未执行”的
        # ToolMessage。之后模型读取 ToolMessage 并决定是否结束或继续。
        interrupts = result.get("__interrupt__")

        if interrupts:
            print("恢复后又产生了新的中断：")
            print(interrupts)
        else:
            print("Agent 已完成，没有产生新的中断。")


def print_usage() -> None:
    """打印 start/resume 两种进程模式的命令格式。"""
    print(
        "用法：\n"
        "  python langgraph_human_in_the_loop_sqlite_demo.py "
        "start <thread_id>\n"
        "  python langgraph_human_in_the_loop_sqlite_demo.py "
        "resume <thread_id> <approve|reject>\n"
    )


def main() -> None:
    """解析命令行参数，选择创建中断或恢复执行。"""
    if len(sys.argv) < 3:
        print_usage()
        return

    mode = sys.argv[1]
    thread_id = sys.argv[2]

    if mode == "start":
        if len(sys.argv) != 3:
            print_usage()
            return

        start_agent(thread_id)
        return

    if mode == "resume":
        if len(sys.argv) != 4:
            print_usage()
            return

        decision = sys.argv[3]
        resume_agent(thread_id, decision)
        return

    print(f"未知模式：{mode}")
    print_usage()

if __name__ == "__main__":
    main()
