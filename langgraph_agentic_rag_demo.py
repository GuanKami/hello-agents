"""实验名称：Agentic RAG 第一版——由模型选择“检索”或“直接回答”。

实验目标：
    在已完成的固定两步 Hybrid RAG 上增加规划节点，观察模型是否能按问题选择
    查询本地知识库，或绕过检索直接回答一般问候。

解决的问题：
    固定 RAG 每次都会检索，即使用户只是问候也会产生不必要的检索；本实验让图
    根据结构化决策走不同分支。

使用的 Agent 概念：
    Agentic RAG、结构化输出、StateGraph 条件路由、Hybrid Retrieval、Grounded
    Generation。

系统架构：
    START -> decide -> (direct -> direct_answer -> END)
                    └-> (retrieve -> grounded_generate -> END)

实现方式：
    decide 节点通过 LLM 输出 RetrievalPlan；Python 条件路由根据 action 选择分支。
    retrieve 分支复用 Dense + BM25 + RRF，随后显式把证据组装进模型上下文；direct
    分支单独回答，不声称查过本地资料。

验证方式：
    维护者提供的真实模型运行结果分别覆盖知识库问题的 retrieve 分支与“你好”的
    direct 分支。运行脚本会调用 Embedding API 和 Chat API，因此可能产生费用。

学习总结：
    模型负责提出路由决策，图只允许执行代码中预先定义的分支；模型的决策并不会
    自动改变图结构。检索结果保存在 State 中，也必须由生成节点显式加入模型消息。

已知限制：
    当前只实现“需要检索 / 直接回答”两种决策，没有证据充分性评估、查询改写、
    有界重试、拒答节点或自动评测。EvidenceReview 与相应 State 字段仅为后续实验
    预留，尚未接入执行流程。向量索引使用 InMemoryVectorStore，进程结束后丢失。
    direct 分支在图运行时绕过 retrieve 查询，但 main() 仍在 invoke 前统一加载语料
    并构建向量索引；因此该路径并未省掉启动阶段的索引构建。
    D# 是单次模型上下文中的临时编号，提示词约束不等于程序化引用校验。

后续优化方向：
    在保持重试次数有上限的前提下，加入证据充分性评估与查询改写；再通过库内、
    库外、改写问法和跨文档问题检查路由、检索与引用质量。
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


class RetrievalPlan(BaseModel):
    """规划节点的受约束输出；它描述动作意图，不直接执行检索。"""

    action: Literal["retrieve", "direct"] = Field(
        description="需要查本地资料时为 retrieve，否则为 direct"
    )
    query: str = Field(
        description="适合用于检索的查询；direct 时可以为空字符串"
    )

class EvidenceReview(BaseModel):
    """后续证据评估实验预留的结构，目前没有节点调用此 Schema。"""

    sufficient: bool = Field(
        description="当前召回片段是否足以回答用户问题"
    )
    retry_query: str | None = Field(
        default=None,
        description="证据不足时给出的新检索查询；不需要重试时为空"
    )

class AgenticRAGState(TypedDict):
    """贯穿图执行的数据；部分字段为后续证据评估 / 重试实验预留。"""

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


def build_graph(vector_store, llm, bm25, chunks):
    """通过依赖注入组装图，避免节点函数依赖 main() 中的隐式全局变量。"""
    # include_raw=True 让调试输出同时保留原始消息、解析结果和解析异常；
    # 不同模型服务对结构化输出的支持可能不同，解析失败时可据原文定位问题。
    planner = llm.with_structured_output(
        RetrievalPlan,
        include_raw=True,
    )

    def decide_node(state: AgenticRAGState):
        """调用规划模型，只产出路由字段，不在此节点执行检索或最终回答。"""
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

        plan = result["parsed"]
        if plan is None:
            # 解析失败时停止图执行，避免用不完整计划误选分支或悄悄退化。
            raise result["parsing_error"]

        # 结构化输出成为 State 更新；真正的分支选择由下面的 Python 条件边完成。
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
        # 引用要求只是提示词约束；当前没有独立的引用事实核验节点。
        response = llm.invoke(
            [
                system_message,
                human_message,
            ]
        )

        return {
            "answer": response.content,
        }

    builder = StateGraph(AgenticRAGState)

    builder.add_node("decide", decide_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("direct_answer", direct_answer_node)
    builder.add_node("grounded_generate", grounded_generate_node)

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
    builder.add_edge("retrieve", "grounded_generate")
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

    # 默认使用知识库内问题；改成“你好”可观察 direct 分支，不需检索资料。
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

    # 图运行时先规划，再根据 action 执行直接回答或检索后生成。
    result = graph.invoke({"question": query})
    print(result["answer"])

if __name__ == "__main__":
    main()
