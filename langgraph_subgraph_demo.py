"""实验名称：Subgraph——将 Agentic RAG 作为父图节点复用。

实验目标：理解编译后的子图如何直接注册为父图节点，以及共享字段的交接。
解决的问题：复用 RAG 流程，避免父图重复实现检索、证据评估和回答逻辑。
使用的 Agent 概念：Subgraph、State schema、共享字段、图组合与职责划分。

系统架构：
    父图：START -> prepare -> rag（编译后的子图）-> format_result -> END。
    子图：decide -> direct_answer，或 retrieve -> assess -> grounded_generate；
    证据不足时有限重试或进入 no_evidence，各结束分支返回 answer。

实现方式：
    prepare 将 user_input 转为 question；父子图共同声明 question 和 answer。
    add_node("rag", rag_graph) 直接注册子图，父图将 answer 转为 final_answer。
    旧版通过包装节点 invoke 子图并映射字段；当前只保留直接注册版本。

验证方式：
    2026-10-13 维护者确认两种接入方式已手动运行。知识问题完成父图准备、
    子图检索生成和父图整理；问候得到直接回答。此次注释维护不重跑 API。

学习总结：
    子图是可复用业务流程，父图负责输入和展示。共享字段是交接接口，
    不代表共享同一个可变字典；子图私有字段不自动成为父图输出字段。

已知限制：
    当前为串行图组合，不是并行或 Multi-Agent。未验证父子图 checkpoint、
    中断恢复和嵌套流式输出。沿用 RAG 引用来源错配与评估解析异常限制。
    main 预先建立向量索引，问候也有 Embedding 初始化成本；向量库仅在
    当前进程有效，真实运行可能产生 Embedding 和 Chat API 费用。

后续优化方向：
    学习并行节点和汇合，再学习 reducer、并行子图与 Multi-Agent。
    将来对照 DeerFlow 实际源码研究流程组合，不将本实验视为其实现。
"""
import os
from dotenv import load_dotenv

from typing import TypedDict, NotRequired

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langchain_core.vectorstores import InMemoryVectorStore

from langgraph_agentic_rag_demo import (
    build_graph as build_rag_graph,
    load_documents,
    split_documents,
)
from langgraph_hybrid_rag_demo import build_bm25

class ParentState(TypedDict):
    """仅声明父图字段及共享接口，不复制 RAG 的全部中间状态。"""
    user_input: str
    # 与子图同名：question 传入问题，answer 接收所有结束分支的结果。
    question: NotRequired[str]
    answer: NotRequired[str]
    final_answer: NotRequired[str]

def build_parent_graph(rag_graph):
    """接收已编译的 RAG 图，构建串行的父级工作流。"""
    def prepare_node(state: ParentState):
        """规范化输入，通过字段更新为子图提供 question。"""
        question = state["user_input"].strip()
        if not question:
            raise ValueError("用户输入不能为空")

        print("[parent.prepare] question =", question)
        return {"question": question}

    def format_result_node(state: ParentState):
        """整理子图结果，不再次调用模型或检索服务。"""
        # answer 由 RAG 子图写回，父图只负责整理最终展示。
        print("[parent.format_result] 整理最终结果")
        return {
            "final_answer": f"学习助手回答：\n{state['answer']}"
        }

    builder = StateGraph(ParentState)

    builder.add_node("prepare", prepare_node)
    # 注册图对象而不是调用 invoke，由框架在运行到此节点时调度子图。
    # documents、retrieval_count 等字段只在子图 schema 中，父图不接收它们。
    builder.add_node("rag", rag_graph)
    builder.add_node("format_result", format_result_node)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "rag")
    builder.add_edge("rag", "format_result")
    builder.add_edge("format_result", END)

    return builder.compile()

def main() -> None:
    """初始化依赖并运行父图；真实执行会访问 Embedding 和 Chat API。"""
    load_dotenv()
    
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

    # 两路检索使用同一批文本块，确保融合时对应的是同一份语料。
    documents = load_documents()
    chunks = split_documents(documents)

    query = "StateGraph 中 State、Node、Edge 分别负责什么？"

    # BM25 在本地建立关键词索引。
    bm25 = build_bm25(chunks)

    # 向量索引需要调用 Embedding API；当前存储只在本进程有效。
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)

    # 复用建图函数并注入依赖，不调用旧实验的 main 来触发额外运行。
    rag_graph = build_rag_graph(
        vector_store, llm, bm25, chunks
    )

    parent_graph = build_parent_graph(rag_graph)

    result = parent_graph.invoke({
        "user_input": query,
    })

    # Mermaid 展示静态拓扑；上面的 invoke 才实际执行工作流。
    print("\n=== 父图 Mermaid 源码 ===")
    print(parent_graph.get_graph().draw_mermaid())

    print("\n=== RAG 子图 Mermaid 源码 ===")
    print(rag_graph.get_graph().draw_mermaid())

    print("\n=== 父图最终回答 ===")
    print(result["final_answer"])
    

if __name__ == "__main__":
    main()


