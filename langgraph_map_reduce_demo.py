"""实验名称：Send 动态分发与 Map-Reduce。

实验目标：按输入主题数量创建任务，观察同一节点的多次并发执行与结果汇合。
解决的问题：固定并行需要预先定义分支；本实验让任务数量由本次输入决定。
使用的 Agent 概念：StateGraph、Send、任务局部输入、Annotated、operator.add reducer。
系统架构：START -> prepare -> Send(explain_topic) × N -> combine -> END。
实现方式：清理并去重主题；每个任务仅接收 topic，返回单项 results；reducer 拼接
    结果列表，combine 按 topic 建立映射，再按清理后的输入顺序拼接最终回答。
验证方式：维护者已提供 State/Node/Edge 三主题真实模型日志：三个任务均开始后，
    按 Node、Edge、State 顺序完成；combine 收到 3 项，按 State、Node、Edge 展示。
    这是一次正常路径手动验证；其他任务数量、空输入和异常路径未运行验收。
学习总结：Send 创建同一个节点的多个任务实例，不是动态添加图节点；任务返回的
    State 更新通过 reducer 汇合，任务完成顺序不决定最终展示顺序。
已知限制：每个去重主题调用一次 Chat API，可能产生费用；无 Embedding 或检索证据。
    模型解释未校验准确性，150 字只是提示词约束；没有显式并发上限、重试、失败降级
    或 checkpoint，不代表生产级批处理或多核 CPU 并行。
后续优化方向：学习 Supervisor / Multi-Agent；按实际需要补充并发限制、失败处理、
    稳定任务 ID 与更广输入验收，保留当前文件作为动态分发的学习基线。
"""

import os
import operator
from dotenv import load_dotenv
from typing import Annotated, NotRequired, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class MapReduceState(TypedDict):
    """全图状态：主题列表、累计任务结果和汇总文本。"""

    topics: list[str]
    # 并发任务写同一个字段时，用列表相加合并；它不负责去重、排序或核验内容。
    results: Annotated[list[dict[str, str]], operator.add]
    final_answer: NotRequired[str]

class TopicTask(TypedDict):
    """Send 传给单个任务的局部输入；不需要接收全图的 topics/results。"""

    topic: str

def build_graph(llm):
    """注入模型并编译图；构图本身不发起模型请求。"""

    def prepare_node(state: MapReduceState):
        # 去除首尾空白与空主题，按首次出现顺序去重，使 topic 可作为结果映射键。
        topics = list(dict.fromkeys(
            topic.strip() for topic in state["topics"] if topic.strip()
        ))
        if not topics:
            raise ValueError("至少提供一个非空主题")
        print("[prepare] topics =", topics)
        return {"topics": topics}

    def dispatch_topics(state: MapReduceState):
        # 路由返回 Send 列表：同一个 explain_topic 节点产生 N 个任务实例。
        # prepare 已拒绝空列表，因此这里至少分发一项任务。
        return [
            Send("explain_topic", {"topic": topic})
            for topic in state["topics"]
        ]

    def explain_topic_node(state: TopicTask):
        # 每个实例收到自己的 topic；等待外部响应时，其他实例可同时推进。
        topic = state["topic"]
        print(f"[explain_topic] 开始：{topic}")
        response = llm.invoke(
            f"请用中文简要解释 LangGraph 中的 {topic}，不超过150字。"
        )
        print(f"[explain_topic] 完成：{topic}")
        # 只返回本任务新增的一项；若携带旧 results，追加型 reducer 会重复累加。
        return {"results": [{"topic": topic, "content": response.content}]}

    def combine_node(state: MapReduceState):
        # 本图各任务只有一层：这一轮任务结束、更新合并后进入汇总节点。
        # Reduce 使用 Python 整理文本，不再调用模型，也不核验生成内容。
        print("[combine] 收集到结果数 =", len(state["results"]))
        by_topic = {
            item["topic"]: item["content"] for item in state["results"]
        }
        # 不依赖 results 列表顺序或完成顺序；prepare 已保证 topic 唯一。
        # 若未来保留重复主题，应改用任务 ID；缺失结果目前会抛 KeyError。
        answer = "\n\n".join(
            f"\n{by_topic[topic]}"
            for topic in state["topics"]
        )
        return {"final_answer": answer}

    builder = StateGraph(MapReduceState)
    builder.add_node("prepare", prepare_node)
    builder.add_node("explain_topic", explain_topic_node)
    builder.add_node("combine", combine_node)
    builder.add_edge(START, "prepare")
    builder.add_conditional_edges(
        "prepare", dispatch_topics, ["explain_topic"]
    )
    # 只注册一个 worker 节点；实际任务数由路由返回的 Send 数量决定。
    builder.add_edge("explain_topic", "combine")
    builder.add_edge("combine", END)
    return builder.compile()

def main() -> None:
    """真实模型手动入口：默认三主题，各请求一次 Chat API，没有持久化状态。"""
    load_dotenv()
    
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )
    graph = build_graph(llm)

    # Mermaid 展示静态拓扑，不展开本次实际分发的三个任务实例。
    print("=== StateGraph Mermaid 源码 ===")
    print(graph.get_graph().draw_mermaid())

    result = graph.invoke({
        "topics": ["State", "Node", "Edge"],
        "results": [],  # 新的一次无 checkpoint 运行，从空列表开始累加。
    })

    print("\n=== Map-Reduce 最终回答 ===")
    print(result["final_answer"])

if __name__ == "__main__":
    main()
