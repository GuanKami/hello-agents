"""实验名称：使用 BM25 实现中文关键词检索。

实验目标：独立观察本地知识库从文本切块、中文分词到关键词 Top-K 排序的完整过程。
解决的问题：向量检索擅长语义相似度，但对精确术语、名称或关键词匹配不一定稳定；本实验提供可对照的词项检索基线。
使用的 Agent 概念：RAG 文档加载、Chunk、稀疏/关键词检索、Top-K 候选召回。
系统架构：Markdown 文件 -> Document -> 文本块 -> jieba 分词 -> BM25Okapi -> 排序结果。
实现方式：使用与向量 RAG 相同的切块参数，避免因分块差异干扰检索方式对比；BM25 分数只表示当前语料内的排序信号。
验证方式：运行本文件，观察三个知识库问题与一个 DeerFlow 无答案问题的 Top-3 结果。
学习总结：BM25 不调用 Embedding 或 Chat API；中文语料需要先分词，检索分数不能直接解释为概率。
已知限制：小型本地语料不能代表真实检索质量；Top-K 会返回候选，即使知识库不包含答案也不会自动拒答。
后续优化方向：与 Dense 检索对比，再用 RRF 融合排名，并为答案缺失设计显式拒答或相关性评估。
"""

import jieba
from rank_bm25 import BM25Okapi

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "rag_docs"


def load_documents() -> list[Document]:
    """读取本实验知识库，并给每份原文记录可回溯的文件名。"""
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
    对比结果才主要反映检索算法差异，而不是分块差异。
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
    """用 jieba 将中文句子切成词项；BM25 根据查询词与文本块词项的重合计分。"""
    return [word.strip() for word in jieba.lcut(text) if word.strip()]

def build_bm25(chunks: list[Document]) -> BM25Okapi:
    """
    根据切分后的 Document 列表构建 BM25 索引。

    BM25 不直接处理 LangChain 的 Document 对象，因此先从每个切块取出
    page_content，再分词并建立词项语料索引。索引与 chunks 的顺序一一对应，
    后续可用命中的索引取回原 Document 及其 source/start_index 元数据。
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
    使用 BM25 对查询进行关键词检索。返回的 score 是 BM25 排序分数，
    不是概率、置信度，也不能与向量检索的相似度分数直接比较。

    返回：
        按 BM25 分数从高到低排列的
        (Document, score) 列表。
    """
    query_tokens = tokenize(query)

    # 对语料中的每个文本块计算与查询词项相关的 BM25 分数。
    scores = bm25.get_scores(query_tokens)

    # 先按分数降序排列文本块索引，再取前 k 个候选；即使查询超出知识库范围，
    # 这里仍会返回分数最高的候选，因此 Top-K 本身不是“有答案”的判断。
    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: scores[index],
        reverse=True,
    )[:k]

    return [
        (chunks[index], float(scores[index]))
        for index in ranked_indices
    ]

def print_results(
    query: str,
    results: list[tuple[Document, float]],
) -> None:
    """打印查询、来源位置和排序分数，以便人工核对命中内容。"""
    print(f"\n=== BM25 检索：{query} ===")

    for rank, (doc, score) in enumerate(results, start=1):
        print(f"\n--- Top {rank} ---")
        print("来源：", doc.metadata.get("source"))
        print("位置：", doc.metadata.get("start_index"))
        print("分数：", f"{score:.4f}")
        print(doc.page_content)

def main() -> None:
    """
    运行纯 BM25 检索实验。

    这个独立实验只观察：
    文档加载 -> 切块 -> 中文分词 -> BM25 排序 -> Top-K。

    暂时不调用 LLM 或 Embedding 服务，先把关键词检索本身看清楚。
    最后一个问题故意询问知识库没有覆盖的 DeerFlow 信息，用来观察 BM25
    仍会给出候选而不会自行拒答这一边界。
    """
    documents = load_documents()
    chunks = split_documents(documents)
    bm25 = build_bm25(chunks)

    questions = [
        "StateGraph 中 State、Node、Edge 分别负责什么？",
        "图里的数据、执行单元和流转关系分别是什么？",
        "ReAct 中 ToolMessage 代表什么？",
        "DeerFlow 使用什么数据库保存长期记忆？",
    ]

    for question in questions:
        results = retrieve_bm25(
            bm25=bm25,
            chunks=chunks,
            query=question,
            k=3,
        )
        print_results(question, results)

if __name__ == "__main__":
    main()
