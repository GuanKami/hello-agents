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
    两类专家均通过 content_and_artifact 返回答案与程序执行摘要。
    知识专家从 RAG State 提取检索次数、各轮证据评估及实际退出节点。
    create_agent 内部组织模型、工具和返回模型的循环。
    evaluate_case 检查专家集合、工具返回 ID、最终回答与文本断言；
    run_evaluation 逐条打印检查并返回报告；save_evaluation_report 保存 JSON。
    数学专家已注册 handle_math_tool_error，错误消息的 artifact 保存错误码。
    ask_math_agent 汇总耗时和工具事件，经外层 artifact 返回父层评估器。
验证方式：
    2026-10-16，维护者提供三条真实运行：知识问题仅调用知识专家；除法仅调用
    数学专家并执行 divide；混合问题调用两个专家后汇总，观察到专家执行重叠。
    2026-10-19，维护者提供批量汇总 3/3，当前流程检查与最小业务断言通过。
    2026-10-20，三用例 JSON 报告 3/3，知识和混合答案均保留 [D2]。
    2026-10-21，寒暄与库外问题 2/2，库外问题两次评估不足后正常退出。
    2026-10-23，supervisor_20261023_103103_117778.json 除零单例 1/1：
    divide 实际收到 a=10、b=0，中间件记录 ValueError / DIVISION_BY_ZERO，
    三项内部执行检查通过；专家耗时 5.17 秒、整例 8.39 秒。
    2026-10-28，supervisor_20261028_104622_109646.json 知识两例 2/2：
    有答案检索一次后生成；库外问题第二轮解析失败，在上限处保守回退退出。
    当前启用知识两例；新增检索上限断言已写入代码，尚无含此断言的新报告。
    历史各批分别通过，不代表当前版本同批六例回归通过。
    历史报告保留限流及评估解析异常；成功复跑不证明异常已修复。
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
    数学专家记录工具事件；知识两例检查检索、评估记录及实际退出节点。
    专家 invoke 抛出未处理异常时没有完整 trace；集合比较不限制重复调用。
    文本断言较粗：State 可命中 StateGraph，224.6 可命中更长数字；
    知识和混合用例正则提取 [D数字]，不验证来源或语义；库外/除零的业务
    文本检查依赖短语，除零与知识两例另有独立的内部执行检查。
    invoke 异常已记录且跳过正常打印；未知用例会抛 ValueError，不再空检查通过。
    消息处理、业务检查与写文件错误不在 invoke 的 try 范围内，仍可能中断整批。
    报告含数学与知识专家局部 trace，不含完整内部消息、来源映射或 Token/成本。
    当前知识检查的 if/else 位于知识分支外，其他用例可能访问未定义的 events；
    两条知识用例通过不能证明其他用例在当前版本仍可执行。
    父层专家工具正常返回与内部 divide 业务失败可以同时成立。
后续优化方向：
    先修正知识验收分支的缩进并回归其他用例，再学习 Token 用量统计；
    来源映射、未处理异常与预算后续补充，保留成功及失败报告。
"""
import os
import re
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from dotenv import load_dotenv
from collections.abc import Callable

from langchain.agents.middleware import wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.messages import AIMessage, ToolMessage

from langgraph_agentic_rag_demo import (
    build_graph as build_rag_graph,
    load_documents,
    MAX_RETRIEVALS,
)
from langgraph_hybrid_rag_demo import (
    build_bm25,
    split_documents,
)

from tools import divide

# 评估规则覆盖六类任务；当前仅启用知识有答案和知识库无答案两例。
# 除零与知识两例有内部执行检查；检索上限断言是报告生成后补入的。
# 专家集合只检查选择范围，不约束调用顺序或重复调用次数。
EVALUATION_CASES = [
    {
        "name": "只问知识",
        "question": "请解释 StateGraph 中 Node 的职责。",
        "expected_tools": {"ask_knowledge_agent"},
    },
    {
        "name": "知识库无答案",
        "question": "DeerFlow 使用什么数据库保存长期记忆？",
        "expected_tools": {"ask_knowledge_agent"},
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

    expert_names = {
        "ask_knowledge_agent": "knowledge_agent",
        "ask_math_agent": "math_agent",
    }

    # 显式提取程序生成的 artifact，避免从最终回答推测内部执行。s
    expert_traces = [
        {
            **message.artifact,
            "parent_tool_call_id": message.tool_call_id,
        }
        for message in tool_messages
        if message.name in expert_names
        and isinstance(message.artifact, dict)
        and message.artifact.get("expert") == expert_names[message.name]
    ]

    # 除零检查工具事件；知识两例检查 RAG State 摘要，不从最终措辞推断流程。
    execution_checks = {}

    if case["name"] == "除零异常":
        divide_events = [
            event
            for trace in expert_traces
            if trace.get("expert") == "math_agent"
            for event in trace.get("tool_events", [])
            if event.get("name") == "divide"
        ]

        execution_checks = {
            "存在数学专家执行记录": bool(expert_traces),
            "实际调用 divide 且参数为 10 和 0": any(
                event["args"].get("a") == 10
                and event["args"].get("b") == 0
                for event in divide_events
            ),
            "除零错误被中间件识别": any(
                event["args"].get("a") == 10
                and event["args"].get("b") == 0
                and event["status"] == "error"
                and event["error_code"] == "DIVISION_BY_ZERO"
                for event in divide_events
            ),
        }

    elif case["name"] in {"只问知识", "知识库无答案"}:
        knowledge_traces = [
            trace
            for trace in expert_traces
            if trace.get("expert") == "knowledge_agent"
        ]

        # 这两条验收用例预期只调用一次知识专家。
        trace = (
            knowledge_traces[0]
            if len(knowledge_traces) == 1
            else {}
        )

        retrieval_count = trace.get("retrieval_count", 0)
        events = trace.get("assessment_events", [])

        execution_checks = {
            "恰有一条知识专家执行记录": len(knowledge_traces) == 1,
            "实际执行过检索": retrieval_count >= 1,
            "检索次数未超过上限": (
                1 <= retrieval_count <= MAX_RETRIEVALS
            ),
            "每轮检索都有评估记录": (
                len(events) == retrieval_count
            ),
        }

    if case["name"] == "只问知识":
        execution_checks.update({
            "最后一轮证据充分": (
                bool(events)
                and events[-1].get("sufficient") is True
            ),
            "经过有证据生成节点": (
                trace.get("exit_reason") == "grounded_generate"
            ),
        })

    else:
        execution_checks.update({
            "没有判定证据充分": (
                bool(events)
                and all(
                    event.get("sufficient") is False
                    for event in events
                )
            ),
            "经过无证据退出节点": (
                trace.get("exit_reason") == "no_evidence"
            ),
        })

    execution_check = {
        "applicable": case["name"] in {
            "除零异常",
            "只问知识",
            "知识库无答案",
        },
        "passed": all(execution_checks.values()),
        "checks": execution_checks,
    }

    actual_tools = {call["name"] for call in calls}

    # 按 ID 集合检查请求/返回闭环及显式 error 状态，不证明业务答案正确，
    # 该父层检查不覆盖内部 divide；后者由 execution_check 单独验证。
    # 集合比较仍不能发现重复 ID 的数量差异。
    call_ids = {call["id"] for call in calls}
    returned_ids = {
        message.tool_call_id
        for message in tool_messages
    }

    # 父层专家正常返回和内部 divide 失败可以同时存在；寒暄空集合闭环也通过。
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
        "passed": (
            routing_ok
            and tool_results_ok
            and final_answer_ok
            and business_check["passed"]
            and execution_check["passed"]
            ),
        "routing_ok": routing_ok,
        "tool_results_ok": tool_results_ok,
        "final_answer_ok": final_answer_ok,
        "expected_tools": sorted(case["expected_tools"]),
        "actual_tools": sorted(actual_tools),
        "expert_call_count": len(calls),
        "elapsed_s": round(perf_counter() - started, 2),
        "answer": last_message.content if final_answer_ok else "",
        "error_type": None,
        "expert_traces": expert_traces,
        "execution_check": execution_check,
    }

def run_evaluation(supervisor) -> list[dict]:
    """顺序运行固定用例，逐条打印检查结果，并返回报告列表。"""
    reports = []

    for case in EVALUATION_CASES:
        print(f"\n=== 验收用例：{case['name']} ===")

        report = evaluate_case(supervisor, case)
        reports.append(report)

        # 异常报告没有 business_check 等正常结果字段。
        # 打印异常后跳过本轮，继续运行下一条用例。
        if report["error_type"]:
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

        # 放在循环内，打印当前这一条用例的业务检查明细。
        print("业务结果检查：", report["business_check"]["checks"])
        print("业务结果通过：", report["business_check"]["passed"])
        print("提取到的引用：", report["business_check"]["citation_ids"])

        if report["execution_check"]["applicable"]:
            print("内部执行检查：", report["execution_check"]["checks"])
            print("内部执行通过：", report["execution_check"]["passed"])

        for trace in report["expert_traces"]:
            print("专家执行记录：", trace)

        print("耗时：", report["elapsed_s"], "秒")
        print("结果：", "PASS" if report["passed"] else "FAIL")

        print("\n=== Agent 最终回答 ===")
        print(report["answer"])

    # 循环结束后只打印整批统计。
    passed_count = sum(report["passed"] for report in reports)

    print("\n=== 批量验收汇总 ===")
    print(f"通过：{passed_count}/{len(reports)}")

    return reports

def check_business_answer(case: dict, answer: str) -> dict:
    """检查固定用例的最小文本要求，不验证引用来源及语义支持度。"""
    # 提取 [D1]、[D2] 等标准标记；[1][2] 不会匹配。
    citation_ids = sorted(set(re.findall(r"\[D[1-9][0-9]*\]", answer)))
    has_citation = bool(citation_ids)

    if case["name"] == "只问知识":
        checks = {
            "包含 Node": "Node" in answer,
            "包含 State 或状态": "State" in answer or "状态" in answer,
            "包含字段更新": (
                "字段更新" in answer
                or "状态更新" in answer
            ),
            "包含标准引用标记": has_citation,
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
            "包含标准引用标记": has_citation,
        }

    elif case["name"] == "寒暄":
        checks = {
            "回答非空": bool(answer.strip()),
            "包含问候": any(
                word in answer
                for word in ("你好", "您好", "嗨", "哈喽")
            ),
            "没有检索引用标记": not has_citation,
        }

    elif case["name"] == "知识库无答案":
        # 只检查是否表达资料不足；仍需人工阅读，确认没有同时补编答案。
        insufficient_phrases = (
            "证据不足",
            "足够证据",
            "资料不足",
            "信息不足",
            "没有找到",
            "未找到",
            "未提及",
            "未说明",
            "无法确定",
            "无法确认",
        )

        checks = {
            "回答非空": bool(answer.strip()),
            "说明资料或证据不足": any(
                phrase in answer
                for phrase in insufficient_phrases
            ),
        }

    elif case["name"] == "除零异常":
        # 只检查错误说明；模型直接回答常识也可能通过，不能替代工具执行证据。
        mentions_zero_division = any(
            phrase in answer
            for phrase in (
                "除数不能为 0",
                "除数不能为0",
                "除数为零",
                "除以零",
                "除以 0",
            )
        )

        explains_failure = any(
            phrase in answer
            for phrase in (
                "无法计算",
                "不能计算",
                "不能为",
                "未定义",
                "没有定义",
                "失败",
            )
        )

        checks = {
            "回答非空": bool(answer.strip()),
            "说明除零原因": mentions_zero_division,
            "说明无法计算": explains_failure,
        }

    else:
        # 新增用例却没有配置断言时，明确报错，避免空检查自动通过。
        raise ValueError(f"尚未配置业务检查：{case['name']}")

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "citation_ids": citation_ids,
    }

def save_evaluation_report(
    reports: list[dict],
    *,
    model_name: str,
    started_at: datetime,
    batch_elapsed_s: float,
) -> Path:
    """保存本批验收记录；仅记录模型名称，不保存密钥或服务地址。"""
    report_dir = Path(__file__).resolve().parent / "evaluation_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 微秒用于区分短时间内连续运行的报告。
    run_id = started_at.strftime("%Y%m%d_%H%M%S_%f")
    report_path = report_dir / f"supervisor_{run_id}.json"

    passed_count = sum(report["passed"] for report in reports)

    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at.isoformat(timespec="seconds"),
        "model_name": model_name,
        "summary": {
            "total": len(reports),
            "passed": passed_count,
            "failed": len(reports) - passed_count,
            "batch_elapsed_s": round(batch_elapsed_s, 2),
        },
        "cases": reports,
    }

    # 报告含两类专家局部 trace，不含完整内部消息、问题、来源映射或 Token/成本。
    # invoke 异常仅记录类型和耗时，尚无异常前的完整执行记录。
    # x 模式避免意外覆盖已经存在的报告。
    with report_path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("\n=== 评估报告已保存 ===")
    print(report_path)

    return report_path

@wrap_tool_call
def handle_math_tool_error(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """将 divide 的已知除零 ValueError 转换为可返回模型的错误消息。

    与旧实验的 ZeroDivisionError 不同，当前 tools.py 主动抛 ValueError。
    已在数学专家 middleware 注册；只处理工具名和错误文本均匹配的异常。
    content 供模型说明失败原因，artifact 供程序读取错误码；其他异常原样抛出。
    返回错误消息表示流程可继续，不代表计算成功。
    """
    tool_call = request.tool_call

    try:
        return handler(request)

    except ValueError as exc:
        # 精确对应 tools.py 当前主动抛出的业务异常。
        is_division_by_zero = (
            tool_call["name"] == "divide"
            and str(exc) == "除数不能为 0"
        )

        if not is_division_by_zero:
            raise

        print(
            "[math_tool_error] 捕获除零异常："
            f"tool={tool_call['name']}, "
            f"args={tool_call['args']}, "
            f"tool_call_id={tool_call['id']}"
        )

        return ToolMessage(
            content=(
                "工具执行失败：除数不能为 0，无法计算。"
                "请说明失败原因，不要重试相同参数，也不要编造数值结果。"
            ),
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            status="error",
            artifact={
                "error_code": "DIVISION_BY_ZERO",
                "error_type": type(exc).__name__,
            },
        )

def build_supervisor(llm, rag_graph):
    # 构建时不执行任务；数学专家有自己的消息输入和工具循环。
    # 主 Agent 不直接持有 divide，只能通过专家工具委派。
    # 显式注册 wrapper，已知除零异常才能转换为带错误码 artifact 的消息。
    math_agent = create_agent(
        model=llm,
        tools=[divide],
        middleware=[handle_math_tool_error],
        name="math_agent",
        system_prompt=(
            "你是计算专家，使用中文回答。"
            "每个除法问题都必须调用 divide 工具，以工具结果为准。"
            "即使除数为 0，也必须调用 divide，让工具报告错误。"
            "工具返回错误后，如实说明失败原因，"
            "不要重试相同参数，不要修改用户数字，不要编造计算结果。"
            "不要处理知识库问题。"
        ),
    )

    # 内部错误消息的 artifact 保存错误码；专家工具的 artifact 返回局部 trace。
    # content 进入父模型上下文，artifact 由程序读取；完整 State 不自动返回。
    @tool(response_format="content_and_artifact")
    def ask_math_agent(question: str) -> tuple[str, dict]:
        """委派除法计算任务；请传入完整的计算问题。"""
        print(f"[math_agent] 开始：{question}")
        started = perf_counter()

        result = math_agent.invoke({
            "messages": [
                {"role": "user", "content": question},
            ],
        })

        # 根据内部 tool_call_id，关联工具请求参数与实际返回结果。
        internal_calls = {
            call["id"]: call
            for message in result["messages"]
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        }

        tool_events = []

        for message in result["messages"]:
            if not isinstance(message, ToolMessage):
                continue

            call = internal_calls.get(message.tool_call_id, {})
            print("[math_agent tool artifact]", message.artifact)
            details = (
                message.artifact
                if isinstance(message.artifact, dict)
                else {}
            )

            tool_events.append({
                "name": message.name or call.get("name"),
                "tool_call_id": message.tool_call_id,
                "args": call.get("args", {}),
                "status": message.status,
                "error_code": details.get("error_code"),
                "error_type": details.get("error_type"),
            })

        handled_business_error = any(
            event["name"] == "divide"
            and event["status"] == "error"
            and event["error_code"] == "DIVISION_BY_ZERO"
            for event in tool_events
        )

        # 仅在专家正常返回后构建；耗时不含 Supervisor 的前后模型调用。
        trace = {
            "expert": "math_agent",
            "elapsed_s": round(perf_counter() - started, 2),
            "tool_events": tool_events,
            "handled_business_error": handled_business_error,
        }

        print("[math_agent trace]", trace)
        print("[math_agent] 完成")

        return result["messages"][-1].content, trace

    @tool(response_format="content_and_artifact")
    def ask_knowledge_agent(question: str) -> tuple[str, dict]:
        """
        查询本地知识库中的 LangGraph、StateGraph、ReAct 知识；
        DeerFlow 实现问题也交由此入口核查，资料不足时返回无法回答。
        """
        print(f"[knowledge_agent] 开始：{question}")
        started = perf_counter()

        # 只传入子任务，父 Agent 的完整会话不会自动进入子图上下文。
        # 每次专家调用显式初始化本次任务的过程字段。
        result = rag_graph.invoke({
            "question": question,
            "retrieval_count": 0,
            "assessment_events": [],
        })

        trace = {
            "expert": "knowledge_agent",
            "elapsed_s": round(perf_counter() - started, 2),
            "retrieval_count": result.get("retrieval_count", 0),
            "assessment_events": result.get("assessment_events", []),
            "exit_reason": result.get("exit_reason"),
        }

        print("[knowledge_agent trace]", trace)
        print("[knowledge_agent] 完成")

        # 父工具执行层会将返回值包装为 ToolMessage；不传回完整子图 State。
        # trace 返回检索轮数、评估事件与退出原因；documents 和 D# 映射仍未返回。
        return result["answer"], trace

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
            "关于 DeerFlow 的实现或存储问题，也必须先调用 ask_knowledge_agent"
            "查询本地资料，不根据模型记忆自行回答。"
            "专家返回资料或证据不足时，最终回答明确说明本地资料不足，"
            "不得继续猜测具体数据库或实现方案。"
            "数学专家报告无法计算或工具执行失败时，"
            "如实转述原因，不自行计算，不重复委派相同任务。"
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
    started_at = datetime.now().astimezone()
    batch_started = perf_counter()

    reports = run_evaluation(supervisor)

    batch_elapsed_s = perf_counter() - batch_started

    save_evaluation_report(
        reports,
        model_name=llm.model_name,
        started_at=started_at,
        batch_elapsed_s=batch_elapsed_s,
    )

if __name__ == "__main__":
    main()
