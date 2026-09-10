from dotenv import load_dotenv
from typing import Literal, Any, Dict, TypedDict
from pydantic import BaseModel
from langchain.tools import tool, ToolRuntime
import os
from serpapi import SerpApiClient

# 加载 .env 文件中的环境变量
load_dotenv()

class Context(BaseModel):
    authority: Literal["admin", "user"]
    user_id: str

# 写入用户信息
class UserInfo(TypedDict, total=False):
    name: str   
    language: str
    favorite_topics: list[str]
    birth_year: int
    sex: Literal["male", "female", "other"]
    identity: str

@tool
def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误：SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn", # 语言代码
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()
        
        # 智能解析：优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"

@tool
def calculate(runtime: ToolRuntime[Context, Any], a: int, b: int) -> int:
    """Add two numbers together."""
    authority = runtime.context.authority
    user_id = runtime.context.user_id   
    # 只有admin用户可以访问加法工具
    if authority != "admin":
        raise PermissionError("User does not have permission to add numbers")
    return a + b

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's sunny in {city}!"

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """
    查询当前用户已经明确保存的长期个人资料。

    当用户询问“我是谁”、自己的姓名、偏好、学习目标，
    或此前保存过哪些资料时，必须使用此工具。

    不需要传入 user_id；
    工具会从 runtime.context.user_id 中获取当前真实用户。
    """
    if runtime.store is None:
        raise RuntimeError("长期记忆 Store 尚未注入。")

    user_id = runtime.context.user_id
    user_info = runtime.store.get(("users",), user_id)

    return str(user_info.value) if user_info else "未知用户"

@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """用于保存/更新用户信息"""
    if runtime.store is None:
        raise RuntimeError("长期记忆 Store 尚未注入。")

    user_id = runtime.context.user_id

    existing_item = runtime.store.get(
        ("users",),
        user_id,
        )

    existing_info = (
        existing_item.value
        if existing_item
        else {}
        )

    merged_info = {
        **existing_info,
        **user_info,
        }

    runtime.store.put(
        ("users",),
        user_id,
        merged_info,
        )

    return "成功保存用户信息"