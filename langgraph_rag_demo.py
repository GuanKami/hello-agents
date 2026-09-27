"""
实验名称：使用 LangGraph StateGraph 实现固定两步向量 RAG
实验目标：跑通本地 Markdown 文档从加载、切分、向量索引、检索到基于证据生成回答的完整链路。
解决的问题：聊天模型本身不会自动读取本地资料；RAG 在生成前先检索相关片段，再把片段显式加入模型上下文。
使用的 Agent 概念：Document 与 metadata、RecursiveCharacterTextSplitter、Embedding、InMemoryVectorStore、State、节点和固定边。
系统架构：START -> retrieve -> generate -> END；State 在两个节点间传递用户问题和召回文档。
实现方式：加载 rag_docs/*.md，保留来源 metadata；按字符切块并生成向量；retrieve 节点召回 Top-K；generate 节点显式拼接证据并调用聊天模型。
验证方式：维护者提供一次端到端运行结果：2 份文档切为 9 个片段，召回 3 个 stategraph.md 片段并生成答案；后续回答出现 [D1]、[D2] 引用，但当前上下文未将编号绑定到片段，引用映射尚未验证。
学习总结：切块和检索结果应先独立检查；State 中保存 Document 不会自动把它们传给 LLM，生成节点必须主动构造上下文。
已知限制：InMemoryVectorStore 不持久化；目前只验证了少量正向问题。一次回答把 Edge 的来源标成 start=0，而更直接的定义位于 start=339。最新输出虽出现 [D1]、[D2]，但上下文块仍只有 source/start；available_citations 只是提示词中的允许编号列表，并未建立编号到 Document 的对应关系，因此不能证明引用有据可查。
后续优化方向：在每个上下文块中加入与 available_citations 同源生成的 [D#] 标签，再核对答案引用与片段内容的对应关系；之后补充改写问题、另一文档问题和无答案问题，再比较 BM25 与混合检索。当前结果不代表完整 RAG 章节或生产级引用校验已完成。
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

ROOT = Path(__file__).resolve().parent
# 使用脚本所在目录定位资料，避免运行时依赖当前 PowerShell 工作目录。
DOCS_DIR = ROOT / "rag_docs"

def load_documents() -> list[Document]:
    """读取 rag_docs 下的 Markdown 文档，并用文件名记录可追溯来源。"""
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
    """在 retrieve 与 generate 两个节点之间传递的数据。"""
    question: str
    documents: NotRequired[list[Document]]
    answer: NotRequired[str]

def build_graph(vector_store, llm):
    """
    创建两步 RAG 状态图：

    START
      -> retrieve：根据问题检索相关文档
      -> generate：把问题和文档交给 LLM 生成答案
      -> END

    与 ReAct 不同，这里没有让模型选择是否检索的条件路由；每个输入都会先检索，再生成。
    """

    def retrieve_node(state: RAGState):
        """检索节点只负责召回，并将 Document 列表作为 State 更新返回。"""
        documents = vector_store.similarity_search(
            state["question"],
            k=3,
        )
        return {"documents": documents}

    def generate_node(state: RAGState):
        """生成节点显式把 State 中的问题和证据片段组装为模型消息。"""
        documents = state.get("documents", [])
        question = state["question"]

        # 将检索结果整理成可读上下文；文档不会因为存在于 State 就自动进入 LLM 请求。
        # source/start 保留原文来源和字符位置，便于人工追溯。
        context = "\n\n".join(
            (
                f"[source={doc.metadata.get('source')}, "
                f"start={doc.metadata.get('start_index', '?')}]\n"
                f"{doc.page_content}"
            )
            for doc in documents
        )

        available_citations = ", ".join(
            f"[D{i}]" for i in range(1, len(documents) + 1)
        ) or "无"

        # available_citations 只告诉模型哪些编号可以写进答案，并没有把编号绑定到具体文档。
        # 目前 context 块没有 [D#] 标签，所以模型即使输出 [D1] / [D2]，也无法据此追溯证据。
        # 修复映射时，必须在上面的每个 context 块中按相同顺序加上 [D1]、[D2] 等标签。
        system_message = SystemMessage(
            content=(
                "你是一个基于资料回答问题的助手。"
                "只能依据本次提供的资料回答。"
                "如果资料不足，请明确说明资料中没有足够信息，不要用背景知识补充。"
                f"本次可引用的片段编号只有：{available_citations}。"
                "回答中的关键结论必须引用对应编号，例如 [D2]。"
                "不要编造本次资料中没有提供的编号。"
            )
        )

        human_message = HumanMessage(
            content=(
                f"用户问题：\n{question}\n\n"
                f"检索到的资料：\n{context}"
            )
        )

        # 这里才是真正调用 Chat Model。
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

    后续验收还应分别添加：改写问题、来自另一份资料的问题，以及资料中没有答案的问题。
    这些问题目前尚未作为完整测试集运行。
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

    # InMemoryVectorStore 适合观察检索机制；索引仅在本进程中存在，脚本退出后不会保留。
    vector_store = InMemoryVectorStore(
        embedding=embeddings,
    )
    # 写入文档时会为切块生成向量；查询时还会为问题生成向量。
    vector_store.add_documents(chunks)

    question = "StateGraph 中 State、Node、Edge 分别负责什么？"

    print("\n=== 独立检索结果 ===")
    # 独立检索用于在调用 LLM 前检查召回质量和排序分数。
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
