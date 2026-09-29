"""
实验名称：使用 LangGraph StateGraph 实现固定两步 Hybrid RAG
实验目标：跑通本地 Markdown 文档从加载、切分、Dense/BM25 混合检索、RRF 排序到基于证据生成回答的完整链路。
解决的问题：聊天模型不会自动读取本地资料；Dense 与 BM25 各自召回候选，RRF 按排名融合后，将最终证据显式放入模型上下文。
使用的 Agent 概念：Document 与 metadata、RecursiveCharacterTextSplitter、Embedding、InMemoryVectorStore、jieba、BM25、RRF、State、节点和固定边。
系统架构：rag_docs -> chunks -> Dense Top-3 + BM25 Top-3 -> RRF Top-3 -> START -> retrieve -> generate -> END；State 在两个节点间传递问题与融合后的文档。
实现方式：加载 rag_docs/*.md 并保留来源 metadata；对共享 chunks 建立 Dense 与 BM25 索引；retrieve 节点调用 hybrid_retrieve 并记录各路排名；generate 节点显式拼接证据、临时编号并调用聊天模型。
验证方式：维护者提供了端到端运行结果：2 份文档切为 9 个片段。针对“ReAct 中 ToolMessage 代表什么？工具返回后 Agent 如何继续？”，RRF Top-1 为 react.md start=363（Dense/BM25 均第 1，分数 0.0328），Top-2 为 react.md start=0（均第 2，分数 0.0323），Top-3 为 stategraph.md start=605（Dense 第 3，BM25 未命中，分数 0.0159）；实际发送给 LLM 的上下文与该顺序一致，回答使用了对应的 [D1]、[D2]、[D3] 编号。
学习总结：Dense 擅长语义相似召回，BM25 擅长词项匹配；RRF 融合排名而非两种检索器量纲不同的原始分数。State 中保存 Document 不会自动把它们传给 LLM，生成节点必须主动构造上下文。
已知限制：InMemoryVectorStore 不持久化；当前只有小型本地语料和有限的端到端问题验证。D# 编号只在本次检索结果中临时分配，不是稳定文档 ID；提示词允许列表不等于程序化引用校验。Top-K/RRF 不判断知识库是否包含答案，也不会自动拒答。
后续优化方向：按需补充改写问法、跨文档问题、库外拒答和引用语义评估；重排、拒答阈值及持久化向量数据库作为后续工程化方向，不属于本次基础实验验收范围。
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from typing import NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.messages import HumanMessage, SystemMessage

from langgraph_hybrid_rag_demo import build_bm25, hybrid_retrieve

ROOT = Path(__file__).resolve().parent
# 使用脚本所在目录定位资料，避免运行时依赖当前 PowerShell 工作目录。
DOCS_DIR = ROOT / "rag_docs"

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

class RAGState(TypedDict):
    """图的共享数据契约：question 是输入，documents 和 answer 是节点产出的数据。"""
    question: str
    documents: NotRequired[list[Document]]
    answer: NotRequired[str]

def build_graph(vector_store, llm, bm25, chunks):
    """
    创建两步 RAG 状态图：

    START
      -> retrieve：根据问题检索相关文档
      -> generate：把问题和文档交给 LLM 生成答案
      -> END

    vector_store 负责 Dense 召回，bm25 与 chunks 负责关键词召回并还原 Document。
    两路使用同一批 chunks，保证融合时的切块和去重标识一致。
    与 ReAct 不同，这里没有让模型选择是否检索的条件路由；每个输入都会先检索，再生成。
    """

    def retrieve_node(state: RAGState):
        """执行 Dense 与 BM25 双路 Top-3 召回，经 RRF 排序后将最终 Document 列表写入 State。"""
        ranked_results = hybrid_retrieve(
            vector_store=vector_store,
            bm25=bm25,
            chunks=chunks,
            query=state["question"],
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

        return {"documents": documents}

    def generate_node(state: RAGState):
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

    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)


    return builder.compile()

def main() -> None:
    """
    用一个正向问题观察完整 RAG 路径。

    当前只演示一个正向问题。后续还应分别检查改写问题、来自另一份资料的问题，
    以及资料中没有答案的问题；这些场景目前尚未作为完整测试集运行。
    """
    load_dotenv()

    # Embedding API 用于文档和查询向量化；Chat API 用于基于召回上下文生成答案。
    # 运行本脚本会访问外部模型服务，注意网络、额度和费用。
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

    documents = load_documents()
    print(f"原始文档数量：{len(documents)}")

    # 保留少量 overlap，降低关键句恰好被切块边界拆开的概率；start_index 用于追溯原文位置。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    print(f"切分后的文档片段数量：{len(chunks)}")

    # 关键词索引与下面的向量库共用同一批 chunks，方便 RRF 按来源和起始位置合并同一片段。
    bm25 = build_bm25(chunks)

    # InMemoryVectorStore 适合观察检索机制；索引仅在本进程中存在，脚本退出后不会保留。
    vector_store = InMemoryVectorStore(
        embedding=embeddings,
    )
    # 写入文档时会为切块生成向量；查询时还会为问题生成向量。
    vector_store.add_documents(chunks)

    question = "ReAct 中 ToolMessage 代表什么？工具返回后 Agent 如何继续？"

    print("\n=== 独立检索结果 ===")
    # 独立检索用于在调用 LLM 前检查召回内容和排序分数；分数不是答案正确率或置信概率。
    # 图内 retrieve_node 随后会再次检索同一问题，因此这里会额外产生一次 query embedding 请求。
    retrieved = vector_store.similarity_search_with_score(
        question,
        k=3,
    )

    for doc, score in retrieved:
        print("来源：", doc.metadata.get("source"))
        print("位置：", doc.metadata.get("start_index"))
        print("分数：", f"{score:.4f}")
        print(doc.page_content)
        print("---")

    graph = build_graph(
        vector_store=vector_store,
        llm=llm,
        bm25=bm25,
        chunks=chunks,
    )

    # invoke 才会真正按 START -> retrieve -> generate -> END 执行编译后的 StateGraph。
    result = graph.invoke(
        {
            "question": question,
        }
    )

    print("\n=== 最终回答 ===")
    print(result["answer"])

    # 再打印一次图运行后的 State，确认 retrieve 节点返回的文档确实被后续节点使用并保留下来。
    print("\n=== State 中保存的召回文档 ===")
    for doc in result.get("documents", []):
        print("来源：", doc.metadata.get("source"))
        print("位置：", doc.metadata.get("start_index"))
        print(doc.page_content)
        print("---")

    # Mermaid 只是图结构的静态描述；实际业务已由上面的 graph.invoke 执行。
    print("\n=== StateGraph Mermaid 源码 ===")
    print(graph.get_graph().draw_mermaid())
    
if __name__ == "__main__":
    main()
