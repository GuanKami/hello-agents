"""实验名称：Agentic RAG 第一版——规划、证据评估与有界检索重试。

实验目标：
    在固定两步 Hybrid RAG 前加入规划节点，并在检索后评估召回证据是否足以回答。
    证据不足时允许改写查询并重试，但总检索次数由程序限制。

解决的问题：
    固定 RAG 对每个输入都检索；而 Top-K / RRF 即使面对知识库外问题也会返回候选。
    本实验增加 direct 分支、证据充分性判断、有界重试和无证据退出路径。

使用的 Agent 概念：
    Agentic RAG、结构化输出、StateGraph 条件路由、Hybrid Retrieval、证据评估、
    查询改写、有界重试、Grounded Generation 和拒答。

系统架构：
    START -> decide -> (direct -> direct_answer -> END)
                    └-> retrieve -> assess
                                      ├ sufficient -> grounded_generate -> END
                                      ├ retry -> apply_retry_query -> retrieve
                                      └ no_evidence -> no_evidence -> END

实现方式：
    decide 节点先用 Python 规则识别独立寒暄并路由到 direct；其他输入调用规划模型改写
    检索查询，但最终 action 固定为 retrieve。retrieve 复用 Dense + BM25 + RRF；assess
    判断证据是否充分，证据不足时可改写查询并在程序限制内重试；充分时进入
    grounded_generate，否则进入 no_evidence。direct 分支独立回答，不声称查过本地资料。

验证方式：
    已通过少量真实模型手动验收：独立问候进入 direct；知识库内 StateGraph 问题及带有
    问候的知识问题进入 retrieve；知识库外 DeerFlow 问题经过两次证据不足判断后进入
    no_evidence。该结果验证固定路由分支，不代表引用质量或整体回答质量通过。回答中
    2026-10-13 维护者提供了引用正向样例及两类负向检查：D99 被程序识别；有效 D3
    不支持 Node 返回字段更新，需人工核对。后续答案仍把来自 D2 的 Edge 原文标为 D1。
    完成的是基础验收方法，不是引用正确性保证。第二轮 assess 解析失败后保守退出
    也已观察，不能记为两轮正常评估。运行会调用外部 API，可能产生费用。

学习总结：
    固定路由策略只允许独立寒暄走 direct，其他输入一律 retrieve；规划模型提供查询改写，
    不再决定是否绕过检索。结构化输出只约束字段形状，不保证证据判断或答案正确。证据
    评估负责判断召回内容是否支持答案，重试与退出由 Python 条件路由和次数上限控制。
    检索结果保存在 State 中，也必须由生成节点显式加入模型消息。

已知限制：
    direct 目前只识别一组固定的独立问候/感谢文本；翻译、改写、总结及依赖多轮对话的
    direct 用例尚未实现。EvidenceReview 仍由模型判断，不能保证稳定或无误；检索次数
    上限只保证停止重试，不保证检索质量。D# 是单次上下文中的临时编号，提示词要求
    不能代替引用校验，且已观察到错配。语料仅有少量本地文档，向量索引使用
    InMemoryVectorStore，进程结束后丢失。direct 虽跳过检索节点，main() 仍会预先加载
    语料并建索引。模型调用等运行时异常仍可能使图失败。

后续优化方向：
    基础路径和引用验收方法阶段性收口，下一主线学习 Subgraph，再学习并行和 Multi-Agent。
    保留引用错配失败样例，后续完善摘录归属检查、逐条支持评估及 assess 异常处理。
"""

import os
import re
from dotenv import load_dotenv
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from langgraph_hybrid_rag_demo import (
    build_bm25,
    split_documents,
    hybrid_retrieve,
)

ROOT = Path(__file__).resolve().parent
# 根据脚本位置定位语料，避免启动时 PowerShell 当前目录不同导致找不到文件。
DOCS_DIR = ROOT / "rag_docs"

# 计数包含第一次检索；值为 2 表示最多执行“首轮检索 + 一次改写重试”。
MAX_RETRIEVALS = 2

class RetrievalPlan(BaseModel):
    """规划节点的受约束输出；它描述动作意图，不直接执行检索。"""

    action: Literal["retrieve", "direct"] = Field(
        description="需要查本地资料时为 retrieve，否则为 direct"
    )
    query: str = Field(
        description="适合用于检索的查询；direct 时可以为空字符串"
    )

class EvidenceReview(BaseModel):
    """assess 节点的结构化输出：判断证据充分性，并在需要时提出新检索词。"""

    sufficient: bool = Field(
        description="当前召回片段是否足以回答用户问题"
    )
    retry_query: str = Field(
        description="证据充分时必须返回空字符串；证据不足时返回新检索查询"
    )

class AgenticRAGState(TypedDict):
    """贯穿图执行的数据；节点只返回局部字段更新，由 LangGraph 合并到 State。"""

    question: str
    action: NotRequired[Literal["retrieve", "direct"]]
    search_query: NotRequired[str]
    documents: NotRequired[list[Document]]
    retrieval_count: NotRequired[int]
    sufficient: NotRequired[bool]
    retry_query: NotRequired[str | None]
    answer: NotRequired[str]

def is_standalone_small_talk(question: str) -> bool:
    """只识别独立的寒暄/感谢，不把带知识问题的句子误判为寒暄。"""
    normalized = question.strip().strip("，。！？!?、 ")
    return normalized in {
        "你好", "你好呀", "您好", "嗨", "哈喽",
        "早上好", "晚上好", "谢谢", "谢谢你", "感谢",
    }

def load_documents() -> list[Document]:
    """加载本地 Markdown，并保留来源元数据供切块、检索和引用定位。"""
    documents = []

    for path in DOCS_DIR.glob("*.md"):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name},
            )
        )

    if not documents:
        # 尽早暴露资料目录为空的问题，避免后续得到看似正常但无证据的回答。
        raise ValueError(f"没有在 {DOCS_DIR} 找到 Markdown 资料")

    return documents

def route_after_decide(state: AgenticRAGState) -> str:
    """把规划节点写入 State 的 action 交给条件边选择已注册的图分支。"""
    return state["action"]

def route_after_assess(state: AgenticRAGState) -> str:
    """根据评估结果和程序级上限，选择生成、有限重试或无证据退出。"""
    if state.get("sufficient", False):
        return "sufficient"

    retry_query = (state.get("retry_query") or "").strip()
    current_query = state.get("search_query", "").strip()

    # 模型可以提出改写词，但是否重试由程序控制，不能让模型触发无限循环。
    if state.get("retrieval_count", 0) >= MAX_RETRIEVALS:
        return "no_evidence"

    # 没有新查询，或新旧查询相同，就停止，避免重复执行同一轮检索。
    if not retry_query or retry_query.casefold() == current_query.casefold():
        return "no_evidence"

    return "retry"

def inspect_citation_ids(answer: str, citation_ids: list[str]) -> dict:
    """检查全文编号存在性；不判断摘录归属、逐条覆盖或语义支持。

    missing_citations 仅表示未匹配到任何 [D数字]，只有无效 D99 时也为 False。
    invalid_ids 才指出允许集合外的编号。报告只供观察，不改变路由或拦截答案。
    """
    allowed = set(citation_ids)
    used = set(re.findall(r"\[D\d+\]", answer))

    return {
        "used_ids": sorted(used),
        "invalid_ids": sorted(used - allowed),
        "missing_citations": not used,
    }

def build_graph(vector_store, llm, bm25, chunks):
    """通过依赖注入组装图，避免节点函数依赖 main() 中的隐式全局变量。"""
    # include_raw=True 让调试输出同时保留原始消息、解析结果和解析异常；
    # 不同模型服务对结构化输出的支持可能不同，解析失败时可据原文定位问题。
    planner = llm.with_structured_output(
        RetrievalPlan,
        include_raw=True,
    )

    # 评估结果使用独立 Schema，避免把“证据是否够用”和“答案怎么表述”混成一个任务；
    # include_raw 便于对照模型原文与 Pydantic 解析结果。
    reviewer = llm.with_structured_output(
        EvidenceReview,
        include_raw=True,
        method="json_mode",
    )

    def decide_node(state: AgenticRAGState):
        """调用规划模型并写入路由字段；检索和最终回答由后续节点执行。"""
        question = state["question"].strip()

        if is_standalone_small_talk(question):
            print("[route policy] effective_action=direct, reason=standalone small talk")
            return {
                "action": "direct",
                "search_query": "",
            }

        result = planner.invoke(
            [
                SystemMessage(
                    content=("你只负责决定是否检索本地知识库，不要回答用户问题本身。"
                    "必须按 RetrievalPlan 输出 action 和 query。"
                    "action 只能是 retrieve 或 direct。"
                    "如果 action 是 direct，query 返回空字符串。"
                    )
                ),
                HumanMessage(content=state["question"]),
            ]
        )

        # 调试时观察模型原文和解析错误；不要只打印解析后的 plan。
        print("[planner raw]", result["raw"].content)
        print("[planner parse error]", result["parsing_error"])

        if result["parsing_error"] is not None:
            raise result["parsing_error"]

        plan = result["parsed"]
        if plan is None:
            # 解析失败时停止图执行，避免用不完整计划误选分支或悄悄退化。
            raise ValueError("规划模型未返回有效的 RetrievalPlan")

        print("[planner parsed]", plan)

        # planner_action 只用于观察模型的原始判断，不直接决定图的执行路径。
        # 本实验采用保守路由策略：独立寒暄已在前面直接进入 direct；
        # 其他请求统一进入 retrieve，确保知识问题不会因模型误判而跳过检索。
        planner_action = plan.action  # 仅用于观察模型原始判断
        effective_query = plan.query.strip() or question

        print(
            f"[route policy] planner_action={planner_action}, "
            "effective_action=retrieve"
        )

        return {
            "action": "retrieve",
            "search_query": effective_query,
        }

    def retrieve_node(state: AgenticRAGState):
        """用规划节点给出的查询执行 Dense + BM25 + RRF，并把排序后的文档写入 State。"""
        # 是否检索由固定路由控制；模型只提出查询文本。复用 Hybrid RAG helper，
        # 将查询改写与实际检索分开，便于观察本轮实际召回了什么。
        ranked_results = hybrid_retrieve(
            vector_store=vector_store,
            bm25=bm25,
            chunks=chunks,
            query=state["search_query"],
            final_k=3,
        )

        for rank, (doc, rrf_score, ranks) in enumerate(ranked_results, start=1):
            print(
                f"[RRF Top {rank}] "
                f"source={doc.metadata.get('source')}, "
                f"start={doc.metadata.get('start_index')}, "
                f"score={rrf_score:.4f}, "
                f"ranks={ranks}"
            )

        # helper 返回 (Document, RRF 分数, 各路排名)。生成节点只需要融合顺序的
        # Document；分数和各路排名已打印用于观察，但不是置信度。
        documents = [document for document, _score, _ranks in ranked_results]

        return {
            "documents": documents,
            "retrieval_count": state.get("retrieval_count", 0) + 1,
        }

    def direct_answer_node(state: AgenticRAGState):
        """回答一般问候等非知识库问题；不读取检索文档，也不伪称查过资料。"""
        response = llm.invoke(
            [
                SystemMessage(
                    content="你是一个友好的助手。请直接回答一般问题或问候；"
                            "不要声称查询过本地知识库。"
                ),
                HumanMessage(content=state["question"]),
            ]
        )

        return {"answer": response.content}

    def assess_node(state: AgenticRAGState):
        """评估当前召回内容能否直接支持回答；不足时生成候选改写查询。"""
        documents = state.get("documents", [])

        # 评估器只看到本轮实际召回的片段及其来源位置，不会自动读取整个知识库。
        evidence = "\n\n".join(
            f"[片段 {i}] source={doc.metadata.get('source')}, "
            f"start={doc.metadata.get('start_index', '?')}\n"
            f"{doc.page_content}"
            for i, doc in enumerate(documents, start=1)
        )

        result = reviewer.invoke([
            SystemMessage(content=(
                "你只评估所给片段是否足以回答问题，不要回答问题本身。"
                "只输出一个合法 JSON 对象，不要输出 YAML、Markdown、代码围栏或解释文字。"
                "对象必须且只能包含 sufficient 和 retry_query 两个字段。"
                '证据充分时输出：{"sufficient": true, "retry_query": ""}。'
                '证据不足时输出：{"sufficient": false, "retry_query": "新的检索查询"}。'
                "不能把主题相关或语义相似当成证据充分；只有片段直接支持答案所需关键事实时，"
                "才能令 sufficient 为 true。"
            )),
            HumanMessage(content=(
                f"用户问题：{state['question']}\n"
                f"当前检索查询：{state['search_query']}\n"
                f"召回片段：\n{evidence}"
            )),
        ])

        print("[assess raw]", repr(result["raw"].content))
        print("[assess parse error]", repr(result["parsing_error"]))

        review = result["parsed"]

        if result["parsing_error"] is not None or review is None:
            if state.get("retrieval_count", 0) >= MAX_RETRIEVALS:
                # 已没有后续检索机会；评估结果不可信时按证据不足处理，避免无依据生成。
                print("[assess fallback] 已达到最大检索次数，保守停止并进入 no_evidence")
                return {
                    "sufficient": False,
                    "retry_query": "",
                }

            if result["parsing_error"] is not None:
                raise result["parsing_error"]

            raise ValueError("证据评估模型未返回有效的 EvidenceReview")

        print(
            f"[assess] sufficient={review.sufficient}, "
            f"retry_query={review.retry_query!r}"
        )

        return {
            "sufficient": review.sufficient,
            "retry_query": review.retry_query,
        }

    def grounded_generate_node(state: AgenticRAGState):
        """将检索证据显式组装为模型消息，再生成带临时来源编号的回答。"""
        documents = state.get("documents", [])
        question = state["question"]

        # D 编号按当前检索顺序临时分配，供回答回指；source/start 用于人工定位原文，
        # 不应把 D 编号当作跨请求稳定 ID。
        citation_ids = []
        context_blocks = []

        for i, doc in enumerate(documents, start=1):
            citation = f"[D{i}]"
            citation_ids.append(citation)

            context_blocks.append(
                f"{citation} source={doc.metadata.get('source')}, "
                f"start={doc.metadata.get('start_index', '?')}\n"
                f"{doc.page_content}"
            )

        context = "\n\n".join(context_blocks)
        # 告知模型本轮可用的临时编号；这只是提示约束，不会验证引用是否真正支持结论。
        available_citations = ", ".join(citation_ids) or "无"

        system_message = SystemMessage(
            content=(
                "你是一个严格依据检索片段回答问题的助手。"
                "只能使用本次提供的资料，不要用背景知识补充资料没有明确说明的事实。"
                "先按问题逐项回答；每个关键事实后都要紧跟直接支持它的片段编号。"
                "优先引用直接定义该事实的片段；概述只能支持概述，示例只能支持示例。"
                "如果某个结论没有直接证据，就删去该结论，或说明资料不足。"
                f"本次可引用的编号只有：{available_citations}。"
                "不要编造编号。"
                "引用编号必须对应上下文块标题中的 [D#]；回答前确认该片段确实支持所述事实。"
                "为了方便验收，每条回答必须包含三个部分："
                "结论：只表达一个可以被资料支持的事实。"
                "引用：使用本轮上下文中的 [D#] 编号。"
                "证据原文：从该编号对应的片段中逐字摘录支持结论的文本。"
                "不要改写或编造证据原文；没有证据的结论不要输出。"
            )
        )

        human_message = HumanMessage(
            content=(
                f"用户问题：\n{question}\n\n"
                f"检索到的资料：\n{context}"
            )
        )

        print("\n=== 实际发送给 LLM 的检索上下文 ===")
        print(context)

        # State 中的 Document 不会自动进入模型上下文，必须在这里显式传给 LLM。
        # 结论、引用、证据原文仍为普通文本格式，没有强制的结构化输出 Schema。
        # 后续只检查编号存在性；摘录归属和语义支持关系仍需人工核对。
        response = llm.invoke(
            [
                system_message,
                human_message,
            ]
        )

        answer = response.content
        # 检查真实模型回答；报告即使发现错误也不拦截或修复返回答案。
        report = inspect_citation_ids(answer, citation_ids)
        print("[citation check]", report)

        return {
            "answer": answer
            }

    def apply_retry_query_node(state: AgenticRAGState):
        """把评估器提出的新查询应用到 State，供下一轮 retrieve 使用。"""
        return {
            "search_query": state["retry_query"].strip(),
            # 清空旧值，避免后续节点误读上一轮的改写查询。
            "retry_query": None,
        }

    def no_evidence_node(state: AgenticRAGState):
        """在重试次数耗尽或没有有效新查询时，给出明确的本地资料范围拒答。"""
        return {
            "answer": "我没有在本地资料中找到足够证据来回答这个问题。"
        }

    builder = StateGraph(AgenticRAGState)

    builder.add_node("decide", decide_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("direct_answer", direct_answer_node)
    builder.add_node("grounded_generate", grounded_generate_node)
    builder.add_node("assess", assess_node)
    builder.add_node("apply_retry_query", apply_retry_query_node)
    builder.add_node("no_evidence", no_evidence_node)

    builder.add_edge(START, "decide")
    # 条件边读取有效 action：独立寒暄 direct，其他输入固定 retrieve。
    # planner 原始 action 只供观察，不直接控制这里的路由。
    builder.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "direct": "direct_answer",
            "retrieve": "retrieve",
        },
    )
    builder.add_edge("direct_answer", END)
    builder.add_edge("retrieve", "assess")

    # 评估器输出只是建议；条件路由同时检查检索次数和查询是否有效，避免无限重试。
    builder.add_conditional_edges(
        "assess",
        route_after_assess,
        {
            "sufficient": "grounded_generate",
            "retry": "apply_retry_query",
            "no_evidence": "no_evidence",
        },
    )

    builder.add_edge("apply_retry_query", "retrieve")
    builder.add_edge("grounded_generate", END)
    builder.add_edge("no_evidence", END)

    # compile() 将 State schema、节点和边组装为可运行图；Mermaid 只展示拓扑，
    # 只有 invoke()/stream() 才会实际执行节点和外部模型调用。
    return builder.compile()

def main() -> None:
    load_dotenv()

    # 两个客户端都使用 OpenAI 兼容配置：Embedding 支持 Dense 检索，Chat 模型
    # 用于规划和最终回答。真实运行会产生外部 API 请求及相应费用。
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        check_embedding_ctx_length=False,
    )

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    # 使用知识库内的 StateGraph 问题观察固定 retrieve 路由、证据评估与生成路径。
    query = "StateGraph 中 State、Node、Edge 分别负责什么？"

    documents = load_documents()
    # 固定 Hybrid RAG 与 Agentic RAG 共用切块和索引流程，便于聚焦比较是否增加规划。
    chunks = split_documents(documents)
    bm25 = build_bm25(chunks)

    # 教学用内存向量库；每次启动重新写入切块，不会跨进程持久化。
    # 注意：索引在图 invoke 前统一构建，所以 direct 分支只跳过查询，不跳过此初始化。
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)

    graph = build_graph(
        vector_store=vector_store,
        llm=llm,
        bm25=bm25,
        chunks=chunks,
    )

    print("=== StateGraph Mermaid 源码 ===")
    print(graph.get_graph().draw_mermaid())

    # 负向样例参照维护者贴出的本轮上下文：D3 不支持 Node 字段更新。
    # 编号存在会通过，人工核对应失败；D 编号每轮变化，不能作为永久规则。
    # 当前保留调用但未打印 report；历史负向日志由维护者提供。
    test_answer = (
        "结论：Node 读取当前 State，并返回需要合并回 State 的字段更新。\n"
        "引用：[D3]"
    )

    report = inspect_citation_ids(
        test_answer,
        ["[D1]", "[D2]", "[D3]"],
    )

    # 图运行时先规划；retrieve 分支还会经历证据评估和有限重试/拒答路由。
    result = graph.invoke({"question": query})
    # answer 汇总所有结束分支；no_evidence 是程序固定文本，其余两支由模型生成。
    print("\n=== Agent 最终回答 ===")
    print(result["answer"])

if __name__ == "__main__":
    main()
