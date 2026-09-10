import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from langgraph.graph import MessagesState
from langgraph.runtime import Runtime

from tools import Context, get_weather, get_user_info, save_user_info

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

def build_store() -> InMemoryStore:
    """创建本次程序运行期间共享的 Store。"""
    store = InMemoryStore()

    store.put(
        ("users",),
        "user_1",
        {
            "name": "派大星",
            "language": "中文",
            "favorite_topics": ["LangGraph", "LangChain"],
            "identity": "从业者",
        },
    )

    store.put(
        ("users",),
        "user_2",
        {
            "name": "猪八戒",
            "language": "English",
            "favorite_topics": ["美食"],
            "identity": "厨师",
        },
    )

    return store

@before_model
def pass_first_user_message_to_middleware(state: MessagesState, runtime: Runtime) -> dict | None:
    """
    这是一个示例中间件，它会在模型调用之前执行。
    它会将用户的第一条消息传递给中间件函数。
    """
    if not state["messages"]:
        return None

    first_user_message = next(
        (msg for msg in state["messages"] if msg.type == "human"), None
    )

    if first_user_message:
        print(f"💬 中间件捕获到用户的第一条消息: {first_user_message.content}")
    return None

@after_model
def log_model_tool_usage(state: MessagesState, runtime: Runtime) -> dict | None:
    """
    这是一个示例中间件，它会在模型调用之后执行。
    它会记录模型调用的工具使用情况。
    """
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]

    tool_calls = getattr(last_message, "tool_calls", None)

    if tool_calls:
        print(f"🛠️ 模型使用了工具: {tool_calls}")

    return None

def build_agent(store: InMemoryStore):
    """根据配置创建 Agent，但不执行用户请求。"""
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    return create_agent(
        model=llm,
        tools=[get_weather, get_user_info, save_user_info],
        middleware=[pass_first_user_message_to_middleware, log_model_tool_usage],
        system_prompt="You are a helpful assistant",
        context_schema=Context,
        store=store,
    )

def print_messages(result: dict) -> None:
    for message in result["messages"]:
        message.pretty_print()

def main() -> None:
    """演示如何使用 langgraph 的中间件功能。"""
    load_dotenv()

    # 同一个程序运行期间只创建一份 Store 和 Agent。
    store = build_store()
    agent = build_agent(store)

    # 用户1请求
    result01 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "你好",
                }
            ]
        },
        context=Context(
            user_id="user_1",
            authority="user",
        )
    )
    print("\n=== 用户1的对话结果 ===")
    print_messages(result01)

    # 用户2请求
    result02 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "北京今天天气怎么样？",
                }
            ]
        },
        context=Context(
            user_id="user_2",
            authority="user",
        )
    )
    print("\n=== 用户2的对话结果 ===")
    print_messages(result02)

if __name__ == "__main__":
    main()