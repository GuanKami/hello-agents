"""
实验名称：Supervisor / Multi-Agent：通过工具接口委派专家

实验目标：
    理解协调者如何选择专家、传入子任务并汇总专家结果。
解决的问题：
    将知识检索与除法计算委派给职责明确的组件，复用现有 RAG 和计算工具。
使用的 Agent 概念：
    create_agent、subagents-as-tools、Tool Calling、专家上下文隔离与嵌套图。
系统架构：
用户问题
    ↓
Supervisor（主 Agent）
    ├── ask_knowledge_agent(question)
    │       ↓
    │   现有 Agentic RAG 图
    │       ↓
    │   返回 answer
    │
    └── ask_math_agent(question)
            ↓
        计算 Agent
            ↓
        divide 工具
            ↓
        返回计算结果
    ↓
Supervisor 整理最终回答

实现方式：
    build_supervisor 闭包保存 RAG 图和计算 Agent，两个 @tool 暴露专家入口。
    父 Agent 仅收到专家答案；create_agent 内部组织模型、工具和返回模型的循环。
    evaluate_case 检查专家集合、工具返回 ID、最终回答与文本断言；
    run_evaluation 记录耗时并返回内存报告列表，尚未保存文件。
验证方式：
    2026-10-16，维护者提供三条真实运行：知识问题仅调用知识专家；除法仅调用
    数学专家并执行 divide；混合问题调用两个专家后汇总，观察到专家执行重叠。
    2026-10-19，维护者提供批量汇总 3/3，当前流程检查与最小业务断言通过。
    main 默认顺序运行 EVALUATION_CASES 三条真实模型用例，不是确定性测试。
    曾出现 Supervisor 汇总丢失引用；强化保留引用的提示词后，本次复跑通过。
学习总结：
    工具接口内部可以运行完整 Agent 或编译子图。多个角色可共用模型客户端，
    职责区别来自提示词、工具和输入上下文，并不要求使用不同模型。
    专家答案与最终答案需分别验收；文本命中、引用标记存在不等于语义正确。
已知限制：
    提示词不是权限边界；D# 为单次检索临时编号，专家返回值没有独立来源映射。
    未配置 checkpoint、统一专家异常转换、超时或显式并发限制。
    recursion_limit 限制父图运行步数，不是所有嵌套调用的 API 总次数预算。
    RAG 引用错配和 assess 解析异常仍可能发生；divide 零除会抛 ValueError。
    启动总会建立内存向量索引，纯计算或寒暄也有 Embedding 初始化请求。
    评估仅看父层消息，不验证内部 divide；集合比较不限制专家重复调用。
    文本断言较粗：State 可命中 StateGraph，224.6 可命中更长数字；
    D1-D3 仅检查标记存在，混合用例未检查引用；未知用例的空断言会通过。
    异常报告缺少 business_check，但打印分支仍访问它，可能再次抛 KeyError；
    汇总后的业务明细只对应最后一条用例。这些路径尚未修复或运行验收。
后续优化方向：
    保存本地报告，修正异常报告与逐用例打印，再补充边界用例、来源和调用预算。
"""
import os

from time import perf_counter
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.messages import AIMessage, ToolMessage

from langgraph_agentic_rag_demo import (
    build_graph as build_rag_graph,
    load_documents,
)
from langgraph_hybrid_rag_demo import (
    build_bm25,
    split_documents,
)

from tools import divide

# 专家集合只检查选择范围，不约束调用顺序或重复调用次数。
EVALUATION_CASES = [
    {
        "name": "只问知识",
        "question": "请解释 StateGraph 中 Node 的职责。",
        "expected_tools": {"ask_knowledge_agent"},
    },
    {
        "name": "只做计算",
        "question": "请计算 2246 除以 10。",
        "expected_tools": {"ask_math_agent"},
    },
    {
        "name": "知识与计算",
        "question": "请解释 StateGraph 中 Node 的职责，并计算 2246 除以 10。",
        "expected_tools": {
            "ask_knowledge_agent",
            "ask_math_agent",
        },
    },
]

def evaluate_case(supervisor, case: dict) -> dict:
    """运行真实模型并生成验收报告；计时不含模型与向量索引初始化。"""
    started = perf_counter()

    try:
        result = supervisor.invoke(
            {
                "messages": [
                    {"role": "user", "content": case["question"]},
                ],
            },
            config={"recursion_limit": 20},
        )
    except Exception as exc:
        # 验收层记录失败，继续执行下一条用例。
        # 这里没有把异常当成业务成功结果。
        return {
            "name": case["name"],
            "passed": False,
            "elapsed_s": round(perf_counter() - started, 2),
            "error_type": type(exc).__name__,
        }

    messages = result["messages"]

    # 当前父会话中的工具调用，指向两个专家入口。
    calls = [
        call
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]

    tool_messages = [
        message
        for message in messages
        if isinstance(message, ToolMessage)
    ]

    actual_tools = {call["name"] for call in calls}

    # 按 ID 集合检查请求/返回闭环及显式 error 状态，不证明业务答案正确，
    # 也看不到专家内部工具消息或重复 ID 的数量差异。
    call_ids = {call["id"] for call in calls}
    returned_ids = {
        message.tool_call_id
        for message in tool_messages
    }

    tool_results_ok = (
        call_ids == returned_ids
        and all(
            message.status != "error"
            for message in tool_messages
        )
    )

    # 必须以没有待执行工具调用的 AIMessage 结束，并包含回答。
    last_message = messages[-1]
    final_answer_ok = (
        isinstance(last_message, AIMessage)
        and not last_message.tool_calls
        and bool(last_message.content)
    )

    # 当前假定 content 为文本，未适配内容块列表等形态。
    business_check = check_business_answer(
        case,
        last_message.content,
    )

    routing_ok = actual_tools == case["expected_tools"]

    return {
        "name": case["name"],
        "business_check": business_check,
        "passed": routing_ok and tool_results_ok and final_answer_ok and business_check["passed"],
        "routing_ok": routing_ok,
        "tool_results_ok": tool_results_ok,
        "final_answer_ok": final_answer_ok,
        "expected_tools": sorted(case["expected_tools"]),
        "actual_tools": sorted(actual_tools),
        "expert_call_count": len(calls),
        "elapsed_s": round(perf_counter() - started, 2),
        "answer": last_message.content if final_answer_ok else "",
        "error_type": None,
    }

def run_evaluation(supervisor) -> list[dict]:
    """顺序运行固定用例，返回内存报告；真实模型结果可能波动。"""
    reports = []

    for case in EVALUATION_CASES:
        print(f"\n=== 验收用例：{case['name']} ===")

        report = evaluate_case(supervisor, case)
        reports.append(report)

        if report["error_type"]:
            # 已知缺陷：异常报告不存在 business_check，此处可能抛 KeyError。
            print("业务结果检查：", report["business_check"]["checks"])
            print("业务结果通过：", report["business_check"]["passed"])
            print("结果：", "PASS" if report["passed"] else "FAIL")
            print("结果：FAIL")
            print("异常类型：", report["error_type"])
            print("耗时：", report["elapsed_s"], "秒")
            continue

        print("预期专家：", report["expected_tools"])
        print("实际专家：", report["actual_tools"])
        print("专家调用次数：", report["expert_call_count"])
        print("专家选择正确：", report["routing_ok"])
        print("工具结果完整：", report["tool_results_ok"])
        print("存在最终回答：", report["final_answer_ok"])
        print("耗时：", report["elapsed_s"], "秒")
        print("结果：", "PASS" if report["passed"] else "FAIL")

        print("\n=== Agent 最终回答 ===")
        print(report["answer"])

    passed_count = sum(report["passed"] for report in reports)

    print("\n=== 批量验收汇总 ===")
    print(f"通过：{passed_count}/{len(reports)}")
    # report 此时是最后一条用例，以下两行不是整批的业务明细。
    print("业务结果检查：", report["business_check"]["checks"])
    print("业务结果通过：", report["business_check"]["passed"])

    return reports

def check_business_answer(case: dict, answer: str) -> dict:
    """固定三条用例的文本冒烟检查，不验证完整语义或引用证据支持度。"""
    checks: dict[str, bool] = {}
    # 子串规则可能误判；新增用例必须同步补充，否则 all([]) 会返回 True。

    if case["name"] == "只问知识":
        checks = {
            "包含 Node": "Node" in answer,
            "包含 State": "State" in answer,
            "包含字段更新": (
                "字段更新" in answer
                or "状态更新" in answer
            ),
            "包含引用": any(
                citation in answer
                for citation in ("[D1]", "[D2]", "[D3]")
            ),
        }

    elif case["name"] == "只做计算":
        checks = {
            "包含计算结果": "224.6" in answer,
            "没有工具错误": "失败" not in answer,
        }

    elif case["name"] == "知识与计算":
        checks = {
            "包含 Node": "Node" in answer,
            "包含计算结果": "224.6" in answer,
        }

    return {
        "passed": all(checks.values()),
        "checks": checks,
    }

def build_supervisor(llm, rag_graph):
    # 构建时不执行任务；数学专家有自己的消息输入和工具循环。
    # 主 Agent 不直接持有 divide，只能通过专家工具委派。
    math_agent = create_agent(
        model=llm,
        tools=[divide],
        name="math_agent",
        system_prompt=(
            "你是计算专家，使用中文回答。"
            "遇到除法问题必须调用 divide 工具，以工具结果为准。"
            "不要处理知识库问题。"
        ),
    )

    @tool
    def ask_knowledge_agent(question: str) -> str:
        """查询本地知识库，回答 LangGraph、StateGraph、ReAct 相关知识问题。"""
        print(f"[knowledge_agent] 开始：{question}")

        # 只传入子任务，父 Agent 的完整会话不会自动进入子图上下文。
        result = rag_graph.invoke({
            "question": question,
        })

        print("[knowledge_agent] 完成")

        # 父工具执行层会将返回值包装为 ToolMessage；不传回完整子图 State。
        # documents、检索轮数和 D# 来源映射未随答案返回。
        return result["answer"]

    @tool
    def ask_math_agent(question: str) -> str:
        """委派除法计算任务；请传入完整的计算问题。"""
        print(f"[math_agent] 开始：{question}")

        # 本次仅提供新的任务消息，没有通过 checkpointer 延续专家会话。
        result = math_agent.invoke({
            "messages": [
                {"role": "user", "content": question},
            ],
        })

        print("[math_agent] 完成")

        # 取专家的最后一条回答，不返回其完整内部工具循环。
        return result["messages"][-1].content

    # 工具名、schema 和 docstring 帮助模型选专家；同轮两个工具可重叠执行，
    # 但不能保证每次都调用两个专家或每次都并发。
    return create_agent(
        model=llm,
        tools=[ask_knowledge_agent, ask_math_agent],
        name="supervisor",
        system_prompt=(
            "你是学习助手的协调者，始终使用中文回答。"
            "你负责理解任务、委派专家并整理最终结果。"
            "LangGraph、StateGraph、ReAct 的知识问题，"
            "必须调用 ask_knowledge_agent，不自行凭记忆回答。"
            "除法计算必须调用 ask_math_agent，不自行计算。"
            "一个问题包含两类任务时，分别委派对应专家。"
            "寒暄可以直接回答。"
            "知识库专家返回证据不足时，如实说明，不补编答案。"
            "整理知识库结果时保留引用编号及其对应结论，不重新分配编号。"
            "知识专家返回带引用的结论时，最终回答必须保留对应的 [D数字] 引用，"
            "可以精简文字，但不得省略引用编号，也不得添加专家未提供的引用。"
        ),
    )

def main() -> None:
    load_dotenv()

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        check_embedding_ctx_length=False,
    )

    # 索引在 invoke 前建立，所有请求均有启动成本；add_documents 请求
    # Embedding API，内存向量索引随当前 Python 进程结束而消失。
    documents = load_documents()
    chunks = split_documents(documents)
    bm25 = build_bm25(chunks)

    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)

    rag_graph = build_rag_graph(
        vector_store=vector_store,
        llm=llm,
        bm25=bm25,
        chunks=chunks,
    )

    supervisor = build_supervisor(llm, rag_graph)

    # 父 Agent 的静态拓扑不会自动展开工具函数内部的专家图。
    print("=== StateGraph Mermaid 源码 ===")
    print(supervisor.get_graph().draw_mermaid())

    # 历史单次请求示例；当前批量入口不使用这个变量。
    question = "请解释 StateGraph 中 Node 的职责，并计算 2246 除以 10。"

    # 索引构建一次；各用例只传新消息，没有 checkpoint 会话延续。
    # 批量运行产生多次 Chat / 查询 Embedding 请求，耗时不含启动索引成本。
    run_evaluation(supervisor)

if __name__ == "__main__":
    main()
