"""
实验名称：Embedding 模型接入与 Store 语义检索最小验证
实验目标：跑通"文本 -> 向量 -> 语义检索"完整链路，作为长期记忆章节的前置实验。
解决的问题：chat 模型只输出文本，无法回答"哪两段话意思相近"；embedding 模型把文本
    映射成定长向量（相似语义 = 相近向量），让"语义相似度"变成可计算的数学问题。
使用的 Agent 概念：OpenAI 兼容 embeddings 端点、LangGraph Store 语义索引（IndexConfig）、
    namespace / key / value 记忆结构。当前全栈统一在 OpenRouter：
    chat 用 minimax/minimax-m3:free（实测支持原生工具调用），embedding 用
    qwen/qwen3-embedding-8b（4096 维，credits 接口实测零扣费）。硅基流动方案已弃用——其零余额账户被
    402 门槛拦截（即使模型定价为 0），教训是"免费"标签背后可能还有账户级限制。
系统架构（不经图和节点，直接隔离观察两个组件本身）：
    文本 --[embeddings]--> 1024 维向量          # 连通性验证
    InMemoryStore(index=...)                    # 开启语义索引
    put()      写入瞬间框架自动将 value 向量化并建立索引
    search()   查询时框架自动将 query 向量化，按余弦相似度排序返回
实现方式：LangChain 的 OpenAIEmbeddings 负责向量生成，LangGraph 的 InMemoryStore
    负责存储和检索。embedding 换供应商后旧向量全部失效（不同模型的向量空间不同），
    必须重新 put 或重跑实验——切换 EMBEDDING_MODEL_ID 前要意识到这一点。
验证方式：运行后观察三点——输出维度是否恒为 1024；search 是否按相关性排序；
    相关条目与无关条目的分差是否明显。
学习总结：embedding 模型负责"算"（生成向量），Store 负责"存和找"；开启 index 后
    读写两侧都不需要手动处理向量。同类比对："OpenAI 兼容"端点仍可能存在
    token 化方式和编码格式上的差异，接入新服务时优先跑一次最小探针。
已知限制：实验记录——关键词 query"谁喜欢学习 LangGraph？"得分虚高（字面匹配所致，
    0.51 vs 0.21）；换成零重叠 query"谁喜欢计算机？"后排序出错且分数跌入噪声区
    （0.185 vs 0.175）。根因已定位：默认索引会嵌入整个 value 的 str() 形式，
    dict 的 JSON 键名属于结构噪声，稀释了语义内容。
后续优化方向：
    - 存储改为自然语言事实句（{"text": "..."}）并用 fields=["text"] 只嵌入事实字段；
      注意 value 缺少该字段时会静默失去语义检索能力；
    - 用"谁喜欢计算机？"和"谁爱吃？"双向对照验收语义检索；
    - 将 namespace 改为按用户划分（("memories", "user_1")），对比全局检索与按用户检索的作用域；
    - 安装 numpy 消除 InMemoryStore 纯 Python 向量运算的性能警告；
    - 清理上方复制自 ReAct 实验、本文件用不到的 import。
"""

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
from langchain.agents import create_agent

# 加载 .env 配置。chat 与 embedding 完全共用同一个 OpenRouter 账号：
# 网关地址和密钥是账号级资源，统一读 LLM_BASE_URL / LLM_API_KEY；
# 只有模型 ID 按通道区分（chat 用 LLM_MODEL_ID，embedding 用 EMBEDDING_MODEL_ID）。
# 注意：若将来某一侧要改用其他服务商，再为它加回独立的 *_BASE_URL / *_API_KEY
load_dotenv()

# 两个非默认参数均为服务适配项："OpenAI 兼容"不等于开箱即用

# 1. 向量模型：专门用于长期记忆
embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),    # 账号级密钥，与 chat 共用
    base_url=os.getenv("LLM_BASE_URL"),  # 与 chat 共用同一上游网关
    # 该上游显式拒绝 token 数组输入（实测报错 "Tokenized input is not supported"），
    # 必须发送原始字符串。注意：encoding_format=float 曾是旧 Nemotron 路由的必需项，
    # 换 qwen3-embedding 后已验证不再需要——适配参数跟着上游走，每次更换都要重新做对照实验
    check_embedding_ctx_length=False,
)

# 2. 聊天模型：继续用于 ReAct Agent
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


# ---------- 第一步：最小连通性探针 ----------
# 一次成功的 embed_query 同时验证：端点可达、鉴权通过、适配参数正确。
# 同一模型的输出维度恒定（本模型实测 2048），此数字即下方 IndexConfig 的 dims 来源
test_vector = embeddings.embed_query(
    "LangGraph 的长期记忆通过向量检索相关信息。"
)

print(type(test_vector))
print(len(test_vector))
print(test_vector[:5])

EMBED_DIM = 4096  # qwen3-embedding-8b 输出 4096 维；必须与上方打印的真实维度一致，否则 search 时才暴露错误

# ---------- 第二步：创建开启语义索引的 Store ----------
# 关键对照：InMemoryStore() 不传 index 时只是普通键值存储，put 的向量数据被忽略，
# search 仅支持前缀过滤；传入 index 后才具备语义检索能力。
# "embed" 接受任何实现了 embed_query/embed_documents 的对象——将来更换 embedding
# 服务只需要替换这一个实例，Store 层代码不动（依赖注入的好处）
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": EMBED_DIM,
    }
)

# ---------- 第三步：写入记忆 ----------
# put(namespace, key, value)：namespace 是分层元组（类似对象存储的目录路径），
# key 在同一 namespace 内必须唯一；value 为可 JSON 序列化的 dict。
# 注意时机：向量化发生在写入瞬间，未经 put 的数据对之后的 search 不可见
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

# ---------- 第四步：语义检索 ----------
# search 会将 query 向量化，对该 namespace（含其子空间）全部条目计算余弦相似度，
# 按 score 从高到低返回；item.score 即相似度分数。
# 局限提醒：本 query 与数据的 favorite_topic 有完全相同的关键词，字符串匹配也会给出
# 相同排名——证明链路可用，但不能证明语义检索生效（见文件头"已知限制"）

# results = store.search(
#     namespace,
#     query="谁喜欢计算机？",
# )

# for item in results:
#     print(item.key, item.value, item.score)



def main():
    agent = create_agent(
        model=llm,
        tools=[get_user_info, save_user_info],
        store=store,
        context_schema=Context,
        system_prompt=(
            "You are a helpful assistant. "
            "When the user asks to view their profile, use get_user_info. "
            "When the user gives their name and asks to save it, use save_user_info."
        ),
    )

    user_context = Context(
        authority="admin",
        user_id="user_3",
    )

    # 第一步：让 Agent 写入长期资料
    print("\n=== 第一步：保存用户资料 ===")
    agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我的名字是海绵宝宝，请记住我的名字。",
                }
            ]
        },
        context=user_context,
    )

    # 第二步：同一个 Store 实例中读取资料
    print("\n=== 第二步：直接读取 Store ===")
    item = store.get(
        ("users",),
        "user_3",
    )
    print(item.value if item else "未找到用户信息")

    # 第三步：让 Agent 通过工具读取资料
    print("\n=== 第三步：让 Agent 查询资料 ===")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我是谁？",
                }
            ]
        },
        context=user_context,
    )

    for message in result["messages"]:
        message.pretty_print()

if __name__ == "__main__":
    main()