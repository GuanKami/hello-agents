"""实验名称：固定并行节点与列表 reducer。

实验目标：理解固定分叉、等待汇合与同一 State 字段的并发更新。
解决的问题：两个独立任务可同时请求模型，结果不能互相覆盖。
使用的 Agent 概念：StateGraph、并发节点、reducer、共享 State、结果汇总。
系统架构：START -> prepare -> explain_concept / give_example -> combine -> END。
实现方式：两个分支各调用一次 LLM，返回带 kind 的单项 results 列表；
    operator.add 拼接列表，combine 按 kind 整理文本，不额外调用模型。
验证方式：真实运行：两个分支先开始，随后完成，
    combine 最后执行；results 同时包含 concept、example。分支耗时 53.83s、
    55.85s，总耗时约 55.85s，观察到模型请求重叠执行。
学习总结：节点只返回自己的更新；reducer 合并并发写入，汇合边控制执行时机。
    按 kind 读取结果，不依赖列表位置或任务完成顺序。
已知限制：固定两个分支；没有 Send、子 Agent、checkpoint 或失败恢复。
    模型生成的解释和代码未经正确性验证，不执行生成代码。kind 重复会被
    汇总字典覆盖，缺失会抛 KeyError；按当前纯文本模型响应处理 content。
    两次 Chat 请求可能产生费用；不涉及 Embedding 或 RAG。
后续优化方向：独立学习 Send 动态派发与 Map-Reduce，再学习并行子图和 Multi-Agent。
"""

import os
import operator
from time import perf_counter
from typing import TypedDict, NotRequired, Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

class ParallelState(TypedDict):
    """共享问题、各分支的结果增量以及最终展示文本。"""
    question: str
    # 同一执行步的两个分支均写 results；显式拼接列表，避免普通字段并发覆盖冲突。
    results: Annotated[list[dict[str, str]], operator.add]
    final_answer: NotRequired[str]

def build_graph(llm):
    """注入模型并构建固定并行图；建图本身不请求模型。"""
    def prepare_node(state: ParallelState):
        question = state["question"].strip()
        if not question:
            raise ValueError("问题不能为空")
        print("[prepare] question =", question)
        return {"question": question}

    def explain_concept_node(state: ParallelState):
        started = perf_counter()
        print("[explain_concept] 开始")

        response = llm.invoke(
            "请用中文简要解释以下概念，不提供代码：\n"
            + state["question"]
        )

        print(f"[explain_concept] 完成，耗时 {perf_counter() - started:.2f}s")
        # 返回本分支的新结果，不回传旧列表，避免 reducer 重复追加历史结果。
        return {
            "results": [
                {
                    "kind": "concept",
                    "content": response.content,
                }
            ]
        }

    def give_example_node(state: ParallelState):
        started = perf_counter()
        print("[give_example] 开始")

        response = llm.invoke(
            "请用中文给出以下问题的最小 Python 代码示例，"
            "并简要说明代码，不重复长篇概念解释：\n"
            + state["question"]
        )

        print(f"[give_example] 完成，耗时 {perf_counter() - started:.2f}s")
        # 与概念分支共享 results，但用 kind 标识职责；示例只展示，不执行。
        return {
            "results": [
                {
                    "kind": "example",
                    "content": response.content,
                }
            ]
        }

    def combine_node(state: ParallelState):
        print("[combine] 两个分支已完成，开始汇总")

        # reducer 负责收集，汇合边负责等待；按标签组织输出，不假设列表顺序。
        results_by_kind = {
            item["kind"]: item["content"]
            for item in state["results"]
        }

        return {
            "final_answer": (
                f"概念解释：\n{results_by_kind['concept']}\n\n"
                f"代码示例：\n{results_by_kind['example']}"
            )
        }

    builder = StateGraph(ParallelState)

    builder.add_node("prepare", prepare_node)
    builder.add_node("explain_concept", explain_concept_node)
    builder.add_node("give_example", give_example_node)
    builder.add_node("combine", combine_node)

    builder.add_edge(START, "prepare")

    # 分叉：两个分支都只依赖 prepare 的结果。
    builder.add_edge("prepare", "explain_concept")
    builder.add_edge("prepare", "give_example")

    # 汇合：等待两个分支都完成，再执行 combine。
    builder.add_edge(
        ["explain_concept", "give_example"],
        "combine",
    )

    builder.add_edge("combine", END)
    return builder.compile()

def main():
    """真实模型手动实验；读取本地配置但不打印凭据。"""
    load_dotenv()

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    graph = build_graph(llm)

    print("=== StateGraph Mermaid 源码 ===")
    # Mermaid 仅展示拓扑，invoke 才执行节点和外部请求。
    print(graph.get_graph().draw_mermaid())

    started = perf_counter()
    result = graph.invoke({
        "question": "LangGraph 中 Node 如何读取 State 并返回字段更新？",
        "results": [],  # 新的一次运行从空结果开始，本实验未配置会话持久化。
    })

    print("\n=== Reducer 合并结果 ===")
    for item in result["results"]:
        print("kind =", item["kind"])

    print(f"\n[total] 总耗时 {perf_counter() - started:.2f}s")
    print("\n=== 并行工作流最终回答 ===")
    print(result["final_answer"])

if __name__ == "__main__":
    main()
