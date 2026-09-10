import os
import sqlite3
from typing import Literal

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from tools import Context, calculate, get_weather, search, get_user_info, save_user_info
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.sqlite import SqliteSaver


# 加载模型配置
load_dotenv()

# 1. 聊天模型：继续用于 ReAct Agent
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

# 2. 向量模型：专门用于长期记忆
embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    # 该服务只接受原始字符串，不接受 LangChain 预处理后的 token ID。
    check_embedding_ctx_length=False,
    # 该服务不支持 OpenAI SDK 默认的 base64 向量编码。
)

test_vector = embeddings.embed_query(
    "LangGraph 的长期记忆通过向量检索相关信息。"
)

print(type(test_vector))
print(len(test_vector))
print(test_vector[:5])

EMBED_DIM = 4096

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": EMBED_DIM,
    }
)

namespace = ("users",)

store.put(
    namespace,
    "user_1",
    {
        "name": "派大星",
        "language": "中文",
        "favorite_topic": "LangGraph",
    },
)

store.put(
    namespace,
    "user_2",
    {
        "name": "猪八戒",
        "language": "中文",
        "favorite_topic": "美食",
    },
)

# 第一版可以只保留 get_weather，跑通后再加入其他工具
available_tools = [get_weather, calculate, get_user_info, save_user_info]
# available_tools = [get_weather, calculate, search]

# 告诉模型有哪些工具可以调用
model_with_tools = llm.bind_tools(available_tools)

# 真正执行工具调用的图节点
tool_node = ToolNode(available_tools)

# # 创建sqlite支持的短期记忆
# checkpointer = SqliteSaver(
#     sqlite3.connect("short-memory.db", check_same_thread=False)
#     )

# 模型节点：根据当前消息决定直接回答，还是调用工具
def model_node(state: MessagesState):
    system_message = SystemMessage(
        content="""
    你是一个带长期用户资料功能的助手。

    工具使用规则：

    1. 当用户提供姓名、语言偏好、学习目标、兴趣等个人资料，
    并要求“保存”“记住”或“更新资料”时，必须调用 save_user_info。

    2. 当用户询问“我是谁？”、“我的名字是什么？”、
    “我保存过什么资料？”、“我的偏好是什么？”时，
    必须调用 get_user_info 查询当前用户的长期资料。

    3. 即使当前 thread_id 是新会话、没有聊天记录，
    也不能直接说“不知道用户是谁”；
    必须先尝试调用 get_user_info 查询长期资料。

    4. 用户主动查询自己此前明确要求保存的资料属于已授权操作，
    可以正常调用 get_user_info。

    5. 用户资料相关回答只能依据工具返回结果生成；
    如果工具返回“未知用户”，再说明尚未保存资料。
    """
    )

    messages = [system_message, *state["messages"]]
    response = model_with_tools.invoke(messages)

    # MessagesState 会自动把这条消息追加到 messages 中
    return {"messages": [response]}


# 条件路由：模型产生工具调用就进入 tools，否则结束
def route_after_model(
    state: MessagesState,
) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "end"


# 创建状态图
builder = StateGraph(
    MessagesState,
    context_schema=Context,
)

# 添加节点
builder.add_node("model", model_node)
builder.add_node("tools", tool_node)

# 配置边
builder.add_edge(START, "model")

builder.add_conditional_edges(
    "model",
    route_after_model,
    {
        "tools": "tools",
        "end": END,
    },
)

# 工具执行完后，再次回到模型节点
builder.add_edge("tools", "model")

# 编译图
graph = builder.compile(
    name="state-react", 
    # checkpointer=checkpointer,
    store=store,
    )

def main():
    print("=== StateGraph Mermaid 源码 ===")
    print(graph.get_graph().draw_mermaid())

    # 本次请求属于哪个真实用户。
    # user_id 用于长期记忆 Store 的读写。
    user_context = Context(
        authority="admin",
        user_id="user_3",
    )

    # 第一个会话：保存用户资料
    read_result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="我的名字是海绵宝宝，请你保存下来。"
                )
            ]
        },
        config={
            "configurable": {
                "thread_id": "memory-write-20260907-v1"
            }
        },
        context=user_context,
    )

    # 第二个会话：保存用户资料
    read_result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="我喜欢学习 LangGraph 和 DeerFlow，请把这两个主题也保存下来。")
            ]
        },
        config={
            "configurable": {
                "thread_id": "memory-write-20260907-v2"
            }
        },
        context=user_context,
    )

    # 第三个会话：保存用户资料
    read_result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="我是2001年出生的，常用中文交流，也请保存。")
            ]
        },
        config={
            "configurable": {
                "thread_id": "memory-write-20260907-v3"
            }
        },
        context=user_context,
    )

    # 第四个会话：查询用户资料
    read_result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="请查询我的全部个人资料。")
            ]
        },
        config={
            "configurable": {
                "thread_id": "memory-write-20260907-v4"
            }
        },
        context=user_context,
    )    

    print("\n=== 读取阶段消息 ===")
    for message in read_result["messages"]:
        message.pretty_print()

if __name__ == "__main__":
    main()