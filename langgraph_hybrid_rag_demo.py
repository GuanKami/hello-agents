"""实验名称：对比 Dense、BM25，并用 RRF 融合检索排名。

实验目标：观察语义向量检索与中文关键词检索各自返回什么，再检查 Reciprocal Rank Fusion（RRF）如何综合两路排名。
解决的问题：单一检索方式可能漏掉语义相近或关键词精确匹配的片段；混合检索尝试利用两路召回的互补性。
使用的 Agent 概念：RAG 文档切块、Dense Retrieval、BM25、混合检索、候选 Top-K、RRF。
系统架构：Markdown -> 共享文本块 -> Dense Top-3 + BM25 Top-3 -> 按 source/start_index 去重并做 RRF -> 融合 Top-3。
实现方式：Dense 和 BM25 使用同一批文本块；RRF 按名次累加 1 / (rrf_constant + rank)，不比较两种检索器量纲不同的原始分数；hybrid_retrieve 同时供 langgraph_rag_demo.py 的 retrieve 节点调用。
验证方式：本文件 main() 独立打印 Dense、BM25、RRF 三组结果；维护者另提供了固定两步 StateGraph 的端到端运行结果，观察到 ToolMessage 问题的 RRF 名次、实际模型上下文和回答引用顺序一致。
学习总结：RRF 融合的是排名而不是 Dense/BM25 原始分数；两路都靠前的文本块通常会获得更高融合排名。
已知限制：本文件的 main() 仍是独立检索对比，不编译 StateGraph、不调用 Chat LLM，也没有拒答阈值；完整的混合 RAG 图位于 langgraph_rag_demo.py。Dense 检索会调用 Embedding API。
后续优化方向：按需扩大端到端问题集，检查改写、跨文档、引用语义和库外拒答；是否引入重排或持久化向量数据库，留待具体需求驱动。
"""

import jieba
from rank_bm25 import BM25Okapi

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import NotRequired, TypedDict

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "rag_docs"


class RAGState(TypedDict):
    """本独立对比脚本未实例化该 State；实际 Hybrid RAG 图的 State 定义在 langgraph_rag_demo.py。"""
    question: str
    documents: NotRequired[list[Document]]
    answer: NotRequired[str]

def load_documents() -> list[Document]:
    """读取本地 RAG 知识库，并为原文保留文件名元数据。"""
    documents = []

    for path in DOCS_DIR.glob("*.md"):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name},
            )
        )

    if not documents:
        raise ValueError(f"没有在 {DOCS_DIR} 找到 Markdown 文档")

    return documents

def split_documents(documents: list[Document]) -> list[Document]:
    """
    使用和向量 RAG 相同的参数切分文档。

    这里必须保持参数一致，这样 BM25 和向量检索使用的是同一批文本片段，
    对比结果才主要反映检索方式差异，而不是切块策略差异。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    print(f"原始文档数量：{len(documents)}")
    print(f"切分后的文档片段数量：{len(chunks)}")

    return chunks

def tokenize(text: str) -> list[str]:
    """使用 jieba 对中文文本分词，供 BM25 建立语料和处理查询。"""
    return [word.strip() for word in jieba.lcut(text) if word.strip()]

def build_bm25(chunks: list[Document]) -> BM25Okapi:
    """
    根据切分后的 Document 列表构建 BM25 索引。

    BM25 不直接处理 LangChain 的 Document 对象，因此先取出每个文本块的
    page_content，再分词建立索引。索引中的语料顺序与 chunks 保持一致。
    """
    corpus_tokens = [
        tokenize(doc.page_content)
        for doc in chunks
    ]

    return BM25Okapi(corpus_tokens)

def retrieve_bm25(
    bm25: BM25Okapi,
    chunks: list[Document],
    query: str,
    k: int = 3,
) -> list[tuple[Document, float]]:
    """
    使用 BM25 对查询进行关键词检索。分数用于当前 BM25 结果内部排序，
    不是概率，也不能与 Dense 检索分数直接比较。

    返回：
        按 BM25 分数从高到低排列的
        (Document, score) 列表。
    """
    query_tokens = tokenize(query)

    # 对每个文本片段计算 BM25 分数。
    scores = bm25.get_scores(query_tokens)

    # 先按照分数降序排列片段索引，再截取 Top-K。
    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: scores[index],
        reverse=True,
    )[:k]

    return [
        (chunks[index], float(scores[index]))
        for index in ranked_indices
    ]

def document_key(doc: Document) -> tuple[str, int]:
    """用来源文件名和原文起始位置标识同一切块，供两路结果去重融合。"""
    return (
        doc.metadata["source"],
        doc.metadata["start_index"],
    )

def hybrid_retrieve(
    vector_store,
    bm25: BM25Okapi,
    chunks: list[Document],
    query: str,
    final_k: int = 3,
):
    """分别执行 Dense/BM25 Top-3 并融合，返回按 RRF 排序的文档、分数和各路名次。

    langgraph_rag_demo.py 的 retrieve 节点调用此函数；本文件 main() 则保留独立
    的分路打印逻辑，便于对照 Dense、BM25 与融合后的排序差异。
    """
    # Dense 与 BM25 各自产生候选排名；它们原始分数的含义和量纲不同，
    # 所以融合阶段只使用排名，不把原始分数混在一起计算。
    dense_docs = vector_store.similarity_search(query, k=3)

    bm25_results = retrieve_bm25(
        bm25=bm25,
        chunks=chunks,
        query=query,
        k=3,
    )
    bm25_docs = [doc for doc, _ in bm25_results]

    return reciprocal_rank_fusion(
        [
            ("Dense", dense_docs),
            ("BM25", bm25_docs),
        ],
        final_k=final_k,
    )

def reciprocal_rank_fusion(
    ranked_lists: list[tuple[str, list[Document]]],
    final_k: int = 3,
    rrf_constant: int = 60,
):
    """依据多路排名执行 Reciprocal Rank Fusion。

    对每路中的每个文档，按从 1 开始的名次累加 `1 / (rrf_constant + rank)`；
    默认常数 60 用于降低单一路排名位置变化的影响。函数同时返回各路名次，
    让学习者能检查某个切块为何排在当前位置。
    """
    scores = {}
    documents_by_key = {}
    ranks_by_key = {}

    for retriever_name, docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = document_key(doc)

            # 两路检索命中同一切块时累加分数；使用原文来源和起始位置作为合并键。
            documents_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1 / (rrf_constant + rank)

            # 保留各路排名，方便检查融合排序为何如此。
            ranks_by_key.setdefault(key, {})[retriever_name] = rank

    # 只根据融合分数排序并截取最终候选。RRF 负责排序，不判断候选是否足以回答问题。
    ranked_keys = sorted(scores, key=scores.get, reverse=True)[:final_k]

    return [
        (documents_by_key[key], scores[key], ranks_by_key[key])
        for key in ranked_keys
    ]

def main() -> None:
    """运行独立的 Dense / BM25 / RRF 对比，不生成最终自然语言答案。"""
    load_dotenv()

    # OpenAI 兼容 Embedding 服务只用于 Dense 检索；本脚本不创建 Chat 模型。
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        check_embedding_ctx_length=False,
    )

    query = "StateGraph 中 State、Node、Edge 分别负责什么？"

    documents = load_documents()
    chunks = split_documents(documents)
    bm25 = build_bm25(chunks)

    # InMemoryVectorStore 只在当前进程存在；add_documents 会为语料建立向量索引。
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)

    # 分别保留两路的 Top-3，既能展示检索器本身，也能将它们交给 RRF 融合。
    dense_docs = vector_store.similarity_search(query, k=3)

    bm25_results = retrieve_bm25(
        bm25=bm25,
        chunks=chunks,
        query=query,
        k=3,
    )
    bm25_docs = [doc for doc, _ in bm25_results]

    print(f"\n=== {query} ===")

    print("\n=== Dense Top-3 ===")
    for rank, doc in enumerate(dense_docs, start=1):
        print(f"\nTop {rank}: {doc.metadata.get('source')} "
            f"位置={doc.metadata.get('start_index')}")
        print(doc.page_content)

    print("\n=== BM25 Top-3 ===")
    for rank, (doc, score) in enumerate(bm25_results, start=1):
        print(f"\nTop {rank}: {doc.metadata.get('source')} "
            f"位置={doc.metadata.get('start_index')} 分数={score:.4f}")
        print(doc.page_content)

    # 使用上面已计算的两路结果做 RRF，避免重复 Dense 检索和额外的查询 Embedding 请求。
    hybrid_results = reciprocal_rank_fusion(
        [
            ("Dense", dense_docs),
            ("BM25", bm25_docs),
        ],
        final_k=3,
    )

    print("\n=== RRF 融合 Top-3 ===")
    for rank, (doc, score, ranks) in enumerate(hybrid_results, start=1):
        print(f"\nTop {rank}: {doc.metadata.get('source')} "
            f"位置={doc.metadata.get('start_index')}")
        print(f"Dense排名={ranks.get('Dense', '未命中')}, "
            f"BM25排名={ranks.get('BM25', '未命中')}, RRF分数={score:.4f}")
        print(doc.page_content)

if __name__ == "__main__":
    main()
