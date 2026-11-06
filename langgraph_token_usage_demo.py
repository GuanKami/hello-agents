"""
实验名称：
    Token 用量观察：模型响应与 Callback 统计对照

实验目标：
    理解一次模型调用的 Token 用量，以及 Callback 如何收集用量。

解决的问题：
    只看模型回答，无法知道本次调用使用了多少输入和输出 Token。

使用的 Agent 概念：
    ChatModel、AIMessage、usage_metadata、Callback。

系统架构：
    用户消息
        -> ChatOpenAI.invoke
        -> AIMessage：回答与本次用量
        -> Callback：收集模型响应中的用量

实现方式：
    使用 UsageMetadataCallbackHandler。
    在 invoke 的 config 中注册回调。
    对照响应自带的用量和回调按模型累计的用量。

验证方式：
    真实调用一次模型。
    检查 input_tokens、output_tokens、total_tokens 是否一致。
    用量缺失时明确输出原因，不把缺失记为零。

学习总结：
    待真实运行后填写。

已知限制：
    依赖模型供应商返回用量。
    当前只验证一次非流式聊天调用，不统计费用、失败请求或 Embedding。
    本地版本的回调需要同时取得 usage_metadata 和 model_name 才会累计。

后续优化方向：
    验证复用回调时的累计行为，再接入 Supervisor 的嵌套调用链。
"""

import json
import os

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_openai import ChatOpenAI


TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
)


def print_json(title: str, value) -> None:
    """打印结构化数据，方便观察字段。"""
    print(f"\n=== {title} ===")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    load_dotenv()

    # 沿用现有实验的环境变量，不打印密钥和接口地址。
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    # 当前统计器只用于下面这一次调用。
    usage_callback = UsageMetadataCallbackHandler()

    response = llm.invoke(
        [
            {
                "role": "user",
                "content": "请用一句中文解释 LangGraph 中 Node 的职责。",
            }
        ],
        config={
            "callbacks": [usage_callback],
        },
    )

    print("\n=== 模型最终回答 ===")
    print(response.content)

    # 这是本次响应自身携带的用量。
    response_usage = response.usage_metadata

    # 本地回调使用响应中的 model_name 作为累计字典的键。
    model_name = response.response_metadata.get("model_name")

    print_json("响应自带的 Token 用量", response_usage)
    print_json("回调累计的 Token 用量", usage_callback.usage_metadata)

    print("\n=== 用量一致性检查 ===")
    print("响应中的模型名：", model_name)

    if response_usage is None:
        print("未完成验收：本次响应没有返回 usage_metadata。")
        print("这表示用量未知，不能当作 Token 用量为 0。")
        return

    if not model_name:
        print("未完成验收：响应缺少 model_name。")
        print("当前本地版本的内置回调无法按模型名累计。")
        return

    callback_usage = usage_callback.usage_metadata.get(model_name)

    if callback_usage is None:
        print("未完成验收：回调没有收集到该模型的用量。")
        return

    # 字段缺失与数值不一致分别处理。
    missing_fields = [
        field
        for field in TOKEN_FIELDS
        if response_usage.get(field) is None
        or callback_usage.get(field) is None
    ]

    if missing_fields:
        print("未完成验收：缺少用量字段：", missing_fields)
        return

    checks = {
        field: response_usage[field] == callback_usage[field]
        for field in TOKEN_FIELDS
    }

    print_json("逐字段检查", checks)
    print("结果：", "PASS" if all(checks.values()) else "FAIL")


if __name__ == "__main__":
    main()