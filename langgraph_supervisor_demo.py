"""
实验名称：Supervisor / Multi-Agent：通过工具接口委派专家

实验目标：
    理解协调者如何选择专家、传入子任务并汇总专家结果。
解决的问题：
    将知识检索与除法计算委派给职责明确的组件，复用现有 RAG 和计算工具。
使用的 Agent 概念：
    create_agent、subagents-as-tools、Tool Calling、专家上下文隔离与嵌套图。
系统架构：
用户问题
    ↓
Supervisor（主 Agent）
    ├── ask_knowledge_agent(question)
    │       ↓
    │   现有 Agentic RAG 图
    │       ↓
    │   返回 answer
    │
    └── ask_math_agent(question)
            ↓
        计算 Agent
            ↓
        divide 工具
            ↓
        返回计算结果
    ↓
Supervisor 整理最终回答

实现方式：
    build_supervisor 闭包保存 RAG 图和计算 Agent，两个 @tool 暴露专家入口。
    父 Agent 仅收到专家答案；create_agent 内部组织模型、工具和返回模型的循环。
验证方式：
    2026-10-16，维护者提供三条真实运行：知识问题仅调用知识专家；除法仅调用
    数学专家并执行 divide；混合问题调用两个专家后汇总，观察到专家执行重叠。
    main 默认保留混合问题，修改 question 可复跑单任务；这些不是自动化评测。
学习总结：
    工具接口内部可以运行完整 Agent 或编译子图。多个角色可共用模型客户端，
    职责区别来自提示词、工具和输入上下文，并不要求使用不同模型。
已知限制：
    提示词不是权限边界；D# 为单次检索临时编号，专家返回值没有独立来源映射。
    未配置 checkpoint、统一专家异常转换、超时或显式并发限制。
    recursion_limit 限制父图运行步数，不是所有嵌套调用的 API 总次数预算。
    RAG 引用错配和 assess 解析异常仍可能发生；divide 零除会抛 ValueError。
    启动总会建立内存向量索引，纯计算或寒暄也有 Embedding 初始化请求。
后续优化方向：
    建立专家选择用例与耗时记录，再补充错误处理、来源传递和调用预算。
"""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph_agentic_rag_demo import (
    build_graph as build_rag_graph,
    load_documents,
)
from langgraph_hybrid_rag_demo import (
    build_bm25,
    split_documents,
)

from tools import divide

def build_supervisor(llm, rag_graph):
    # 构建时不执行任务；数学专家有自己的消息输入和工具循环。
    # 主 Agent 不直接持有 divide，只能通过专家工具委派。
    math_agent = create_agent(
        model=llm,
        tools=[divide],
        name="math_agent",
        system_prompt=(
            "你是计算专家，使用中文回答。"
            "遇到除法问题必须调用 divide 工具，以工具结果为准。"
            "不要处理知识库问题。"
        ),
    )

    @tool
    def ask_knowledge_agent(question: str) -> str:
        """查询本地知识库，回答 LangGraph、StateGraph、ReAct 相关知识问题。"""
        print(f"[knowledge_agent] 开始：{question}")

        # 只传入子任务，父 Agent 的完整会话不会自动进入子图上下文。
        result = rag_graph.invoke({
            "question": question,
        })

        print("[knowledge_agent] 完成")

        # 父工具执行层会将返回值包装为 ToolMessage；不传回完整子图 State。
        # documents、检索轮数和 D# 来源映射未随答案返回。
        return result["answer"]

    @tool
    def ask_math_agent(question: str) -> str:
        """委派除法计算任务；请传入完整的计算问题。"""
        print(f"[math_agent] 开始：{question}")

        # 本次仅提供新的任务消息，没有通过 checkpointer 延续专家会话。
        result = math_agent.invoke({
            "messages": [
                {"role": "user", "content": question},
            ],
        })

        print("[math_agent] 完成")

        # 取专家的最后一条回答，不返回其完整内部工具循环。
        return result["messages"][-1].content

    # 工具名、schema 和 docstring 帮助模型选专家；同轮两个工具可重叠执行，
    # 但不能保证每次都调用两个专家或每次都并发。
    return create_agent(
        model=llm,
        tools=[ask_knowledge_agent, ask_math_agent],
        name="supervisor",
        system_prompt=(
            "你是学习助手的协调者，始终使用中文回答。"
            "你负责理解任务、委派专家并整理最终结果。"
            "LangGraph、StateGraph、ReAct 的知识问题，"
            "必须调用 ask_knowledge_agent，不自行凭记忆回答。"
            "除法计算必须调用 ask_math_agent，不自行计算。"
            "一个问题包含两类任务时，分别委派对应专家。"
            "寒暄可以直接回答。"
            "知识库专家返回证据不足时，如实说明，不补编答案。"
            "整理知识库结果时保留引用编号及其对应结论，不重新分配编号。"
        ),
    )

def main() -> None:
    load_dotenv()

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        check_embedding_ctx_length=False,
    )

    # 索引在 invoke 前建立，所有请求均有启动成本；add_documents 请求
    # Embedding API，内存向量索引随当前 Python 进程结束而消失。
    documents = load_documents()
    chunks = split_documents(documents)
    bm25 = build_bm25(chunks)

    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)

    rag_graph = build_rag_graph(
        vector_store=vector_store,
        llm=llm,
        bm25=bm25,
        chunks=chunks,
    )

    supervisor = build_supervisor(llm, rag_graph)

    # 父 Agent 的静态拓扑不会自动展开工具函数内部的专家图。
    print("=== StateGraph Mermaid 源码 ===")
    print(supervisor.get_graph().draw_mermaid())

    question = "请解释 StateGraph 中 Node 的职责，并计算 2246 除以 10。"

    # 默认混合问题会产生多次 Chat 请求；三条验收用例见文件头。
    result = supervisor.invoke(
        {
            "messages": [
                {"role": "user", "content": question},
            ],
        },
        config={"recursion_limit": 20},
    )

    # 父会话包含专家工具调用及结果；专家内部执行过程由各组件另行打印。
    print("\n=== Supervisor 会话消息 ===")
    for message in result["messages"]:
        message.pretty_print()

    print("\n=== Supervisor 最终回答 ===")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
