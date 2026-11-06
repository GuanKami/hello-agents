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

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    # 两次调用共用这个实例，因此用量会累计到同一个回调中。
    usage_callback = UsageMetadataCallbackHandler()

    questions = [
        "请用一句中文解释 LangGraph 中 Node 的职责。",
        "请用一句中文解释 LangGraph 中 State 的职责。",
    ]

    responses = []

    for index, question in enumerate(questions, start=1):
        response = llm.invoke(
            [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            config={
                "callbacks": [usage_callback],
            },
        )

        responses.append(response)

        print(f"\n========== 第 {index} 次调用 ==========")

        print("\n=== 模型最终回答 ===")
        print(response.content)

        # 当前 response 只保存本次调用的用量。
        print_json(
            f"第 {index} 次响应自带的 Token 用量",
            response.usage_metadata,
        )

        # 同一个回调已经收集了截至当前调用的累计用量。
        print_json(
            f"第 {index} 次调用后的回调累计用量",
            usage_callback.usage_metadata,
        )

    print("\n=== 两次调用累计验收 ===")

    # 先检查两次响应是否都提供了模型名。
    model_names = [
        response.response_metadata.get("model_name")
        for response in responses
    ]

    if any(not name for name in model_names):
        print("未完成验收：至少一次响应缺少 model_name。")
        return

    # 本实验先验证同一个模型的两次调用累计。
    if len(set(model_names)) != 1:
        print("未完成验收：两次响应返回的模型名不同。")
        print("模型名：", model_names)
        print("需要按各自模型名分别验收，不能直接比较一个模型的累计值。")
        return

    model_name = model_names[0]

    response_usages = [
        response.usage_metadata
        for response in responses
    ]

    if any(usage is None for usage in response_usages):
        print("未完成验收：至少一次响应没有返回 usage_metadata。")
        print("缺失用量不能当作 0。")
        return

    callback_usage = usage_callback.usage_metadata.get(model_name)

    if callback_usage is None:
        print("未完成验收：回调没有收集到该模型的用量。")
        return

    # 检查字段完整性，再做加法，避免把缺失字段默认成零。
    missing_fields = [
        f"第 {index} 次响应缺少 {field}"
        for index, usage in enumerate(response_usages, start=1)
        for field in TOKEN_FIELDS
        if usage.get(field) is None
    ]

    missing_fields.extend(
        f"回调累计值缺少 {field}"
        for field in TOKEN_FIELDS
        if callback_usage.get(field) is None
    )

    if missing_fields:
        print_json("缺失字段", missing_fields)
        print("未完成验收：用量字段不完整。")
        return

    # 根据两次独立响应，计算预期累计值。
    expected_usage = {
        field: sum(
            usage[field]
            for usage in response_usages
        )
        for field in TOKEN_FIELDS
    }

    # 从回调中取出相同的三个核心字段。
    actual_usage = {
        field: callback_usage[field]
        for field in TOKEN_FIELDS
    }

    checks = {
        field: expected_usage[field] == actual_usage[field]
        for field in TOKEN_FIELDS
    }

    print("本次验收的模型：", model_name)
    print_json("两次响应用量之和（预期）", expected_usage)
    print_json("回调累计用量（实际）", actual_usage)
    print_json("逐字段检查", checks)

    print(
        "\n结果：",
        "PASS" if all(checks.values()) else "FAIL",
    )

        # 保存原回调的三个累计值，便于核对它没有被新实例影响。
    original_totals = {
        field: usage_callback.usage_metadata[model_name][field]
        for field in TOKEN_FIELDS
    }

    # 新建一个独立实例；没有接收模型调用，因此初始统计应为空。
    fresh_callback = UsageMetadataCallbackHandler()

    print_json(
        "新建回调的初始用量",
        fresh_callback.usage_metadata,
    )

    print_json(
        "原回调仍保留的累计用量",
        usage_callback.usage_metadata,
    )

    independence_checks = {
        "新旧回调是不同实例": (
            fresh_callback is not usage_callback
        ),
        "新回调的统计为空": (
            fresh_callback.usage_metadata == {}
        ),
        "原回调的累计值保持不变": all(
            usage_callback.usage_metadata[model_name][field]
            == original_totals[field]
            for field in TOKEN_FIELDS
        ),
    }

    print_json(
        "回调实例独立性检查",
        independence_checks,
    )

    print(
        "\n回调实例独立性结果：",
        "PASS" if all(independence_checks.values()) else "FAIL",
    )

if __name__ == "__main__":
    main()