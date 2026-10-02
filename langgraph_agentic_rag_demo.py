"""3. 第一版只做“需要检索 / 直接回答”中发现了问题：direct 分支当前也进入只允许依据检索片段回答的 generate_node，但此时没有检索片段。"""
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
# 使用脚本所在目录定位资料，避免运行时依赖当前 PowerShell 工作目录。
DOCS_DIR = ROOT / "rag_docs"

class RetrievalPlan(BaseModel):
    action: Literal["retrieve", "direct"] = Field(
        description="需要查本地资料时为 retrieve，否则为 direct"
    )
    query: str = Field(
        description="适合用于检索的查询；direct 时可以为空字符串"
    )

class EvidenceReview(BaseModel):
    sufficient: bool = Field(
        description="当前召回片段是否足以回答用户问题"
    )
    retry_query: str | None = Field(
        default=None,
        description="证据不足时给出的新检索查询；不需要重试时为空"
    )

class AgenticRAGState(TypedDict):
    question: str
    action: NotRequired[Literal["retrieve", "direct"]]
    search_query: NotRequired[str]
    documents: NotRequired[list[Document]]
    retrieval_count: NotRequired[int]
    sufficient: NotRequired[bool]
    retry_query: NotRequired[str | None]
    answer: NotRequired[str]   

def load_documents() -> list[Document]:
    """读取 Markdown 原文，并把来源文件名放进 metadata 供切块后追溯。"""
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
    return state["action"]

def build_graph(vector_store, llm, bm25, chunks):
    # llm 是 build_graph 的参数，因此这里可以使用它
    planner = llm.with_structured_output(
        RetrievalPlan,
        include_raw=True,
        )

    def decide_node(state: AgenticRAGState):
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
            raise result["parsing_error"]

        return {
            "action": plan.action,
            "search_query": plan.query.strip(),
        }

    def retrieve_node(state: AgenticRAGState):
        """执行 Dense 与 BM25 双路 Top-3 召回，经 RRF 排序后将最终 Document 列表写入 State。"""
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

        # hybrid_retrieve 返回 (Document, RRF分数, 各路排名)。
        # State 和后续 generate_node 只需要按最终融合顺序排列的 Document。
        documents = [document for document, _score, _ranks in ranked_results]

        return {
            "documents": documents,
            "retrieval_count": state.get("retrieval_count", 0) + 1
            }

    def direct_answer_node(state: AgenticRAGState):
        """处理不需要查询本地知识库的问题，不要求引用检索片段。"""
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
        """生成节点显式把 State 中的问题和证据片段组装为模型消息。"""
        documents = state.get("documents", [])
        question = state["question"]

        # D 编号仅对本次检索结果有效：按检索排序依次绑定到对应 Document，便于答案回指证据。
        # source/start 是原文定位信息；即使检索顺序变化，它们也能帮助人工找到原始片段。
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
        # 允许列表与上面相同的 documents 数量和顺序生成，只限制可用编号，不会自动校验引用语义。
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

        # State 中的 Document 不会自动进入模型上下文；这里显式发送问题和检索片段。
        # 提示词要求引用只能提供行为约束，模型生成后仍需检查编号是否支持对应结论。
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

    return builder.compile()

def main() -> None:
    load_dotenv()

    # OpenAI 兼容 Embedding 服务只用于 Dense 检索；本脚本不创建 Chat 模型。
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

    query = "StateGraph 中 State、Node、Edge 分别负责什么？"
    
    documents = load_documents()
    chunks = split_documents(documents)
    bm25 = build_bm25(chunks)

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

    result = graph.invoke({"question": query})
    print(result["answer"])

if __name__ == "__main__":
    main()
