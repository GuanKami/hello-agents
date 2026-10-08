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
    decide 节点通过 RetrievalPlan 输出 action 和首轮 query；当前 action 由模型决定，
    Python 条件边只负责把它映射到预先注册的分支。retrieve 节点复用 Dense + BM25 +
    RRF；assess 节点用 EvidenceReview 判断证据充分性并提出 retry_query；Python 路由
    限制总检索次数最多为 MAX_RETRIEVALS。证据充分时才进入 grounded_generate，达到
    上限或没有有效新查询时进入 no_evidence。direct 分支独立回答，不声称查过本地资料。

验证方式：
    维护者提供的真实模型输出观察到问候进入 direct；知识库内问题进入 retrieve，
    EvidenceReview 判为 sufficient；知识库外 DeerFlow 问题曾被规划模型误判为 direct，
    另一次运行则完成两次检索、两次证据不足后进入 no_evidence。知识库内回答出现过
    D# 引用错配，因此引用语义尚未通过验收。这些是少量手动观察，不是自动化评测。
    运行会调用 Embedding API 和 Chat API，可能产生费用。

学习总结：
    结构化输出只约束字段形状，不保证路由判断正确；当前程序尚未用规则校验 direct
    是否只用于寒暄。证据评估负责判断召回内容是否支持答案，是否继续检索则由 Python
    条件路由和次数上限控制。检索结果保存在 State 中，也必须由生成节点显式加入模型消息。

已知限制：
    planner 仍可能把知识库外事实问题误判为 direct，导致无证据的自由回答；确定性的
    direct 策略保护尚未实现。EvidenceReview 也是模型判断，不能保证评估稳定或无误。
    达到上限时只保证停止重试，不保证检索质量；模型调用等运行时异常仍可能使图失败。
    D# 是单次模型上下文中的临时编号，提示词约束不等于程序化引用校验；已观察到引用
    错配。语料仅有少量本地文档，向量索引使用 InMemoryVectorStore，进程结束后丢失。
    direct 分支虽跳过图中的检索节点，但 main() 在 invoke 前仍统一加载语料并建索引。

后续优化方向：
    为 direct 分支增加明确、可观察的程序策略，并用库内、库外、改写问法和跨文档问题
    验收路由与证据评估；再独立检查引用是否真正支持对应结论，并诊断评估节点异常。
"""

import os
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

        # 当前将模型给出的 action 原样写入 State；Python 条件边负责映射分支，
        # 但尚未用确定性规则限制 direct 只能用于寒暄，因此知识问题可能被误路由。
        return {
            "action": plan.action,
            "search_query": plan.query.strip(),
        }

    def retrieve_node(state: AgenticRAGState):
        """用规划节点给出的查询执行 Dense + BM25 + RRF，并把排序后的文档写入 State。"""
        # 这一版只让 Agent 决定“是否检索”和首轮查询文本；检索仍复用已验证的
        # Hybrid RAG helper，而不是让模型自行调用任意检索工具。
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
        # 限定模型可用的编号集合，但不能程序化证明某编号确实支持相邻结论。
        available_citations = ", ".join(
            f"[D{i}]" for i in range(1, len(documents) + 1)
        ) or "无"

        system_message = SystemMessage(
            content=(
                "你是一个严格依据检索片段回答问题的助手。"
                "只能使用本次提供的资料，不要用背景知识补充资料没有明确说明的事实。"
                "先按问题逐项回答；每个关键事实后都要紧跟直接支持它的片段编号。"
                "优先引用直接定义该事实的片段；概述只能支持概述，示例只能支持示例。"
                "例如，片段只说节点返回字段更新，就不要据此断言 State 是可变或不可变对象。"
                "如果某个结论没有直接证据，就删去该结论，或说明资料不足。"
                f"本次可引用的编号只有：{available_citations}。"
                "不要编造编号。"
                "引用编号必须按上下文块标题中的 [D#] 对应，不能凭检索顺序或记忆猜编号；回答前逐条确认被引用的块确实支持该事实。"
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
        # 引用编号约束仍只是提示词；当前没有独立节点验证引用是否支持相邻结论。
        response = llm.invoke(
            [
                system_message,
                human_message,
            ]
        )

        return {
            "answer": response.content,
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
    # LLM 只能在两个预先声明的 action 中选择；模型提出决策，程序控制允许的流程。
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
    builder.add_edge("grounded_generate", END)

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

    # 当前用知识库外的 DeerFlow 问题观察规划与拒答路径；模型也可能误选 direct，
    # 因而该样例同时用于暴露当前尚未解决的路由不稳定问题。
    query = "DeerFlow 使用什么数据库保存长期记忆？"

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

    # 图运行时先规划；retrieve 分支还会经历证据评估和有限重试/拒答路由。
    result = graph.invoke({"question": query})
    print(result["answer"])

if __name__ == "__main__":
    main()
