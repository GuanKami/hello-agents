from tools import search, calculate, get_weather, Context
import os
from dotenv import load_dotenv
from IPython.display import HTML, display, update_display
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.sqlite import SqliteSaver

# 加载模型配置
_ = load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

# 创建短期记忆
checkpointer = InMemorySaver()

# 创建带工具调用的Agent
tool_agent = create_agent(
    model=llm,
    tools=[search, calculate, get_weather],
    system_prompt="You are a helpful assistant",
    context_schema=Context,
    checkpointer=checkpointer
)

# 在运行Agent时注入context
response = tool_agent.invoke(
    {"messages": [{"role": "user", "content": "你好，我是猪八戒"}]},
    config={"configurable": {"thread_id": "1"}},
    context=Context(authority="admin"),
)

for message in response["messages"]:
    message.pretty_print()

result = tool_agent.invoke(
    {"messages": [{"role": "user", "content": "我是谁？"}]},
    config={"configurable": {"thread_id": "1"}},
    context=Context(authority="admin"),
)

for message in result["messages"]:
    message.pretty_print()