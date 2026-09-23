"""实验名称：FastMCP 真实天气服务端

实验目标：把外部天气 HTTP API 封装成 MCP Client 可发现、可调用的工具。
解决的问题：将天气数据获取从 Agent 进程中解耦，避免 Agent 直接依赖供应商 SDK 或服务端函数。
使用的 Agent 概念：MCP Server、stdio、工具 Schema、外部 API 集成。
系统架构：MCP Client -> stdio -> FastMCP -> 城市地理查询 -> 坐标天气查询 -> 工具文本结果。
实现方式：FastMCP 注册 get_weather；函数使用 httpx 请求天气服务，并将选取后的字段格式化返回。
验证方式：当前文件本次只做静态检查，没有发起真实天气 API 请求；实时结果需由维护者手动验证。
学习总结：Server 负责提供工具能力，调用时机由连接它的 Agent 决定；stdio stdout 保留给协议消息。
已知限制：仅查询中国城市；同名候选取第一个；无自动重试；错误 JSON / 字段缺失没有统一处理。
后续优化方向：验证实际响应 Schema 和单位，补足状态码处理、可控重试、日志与预测天气接口。
"""

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

# server.py 位于 mcp_server/get_weather_mcp/ 下，向上两级得到仓库根目录，
# 因而无论从哪个工作目录启动，都从同一位置加载本地 .env。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# FastMCP 会把下面带 @mcp.tool 的函数注册成可由 MCP Client 发现和调用的工具。
# 这里负责提供能力，不负责决定 Agent 何时调用它。
mcp = FastMCP("get_weather_mcp")


@mcp.tool
def get_weather(city: str) -> str:
    """查询中国城市的实时天气，返回天气状况、气温、体感温度和湿度。"""
    api_host = os.getenv("QWEATHER_API_HOST")
    api_key = os.getenv("QWEATHER_API_KEY")

    if not api_host or not api_key:
        # 将配置缺失转为可读的工具结果，避免 MCP 子进程启动后因缺少配置直接崩溃。
        return "天气服务未配置，请检查 .env 中的 QWEATHER_API_HOST 和 QWEATHER_API_KEY。"

    # Host 由本地配置提供（只填域名，scheme 由代码补上），密钥放在请求头中；不要拼进 URL 或日志。
    headers = {"X-QW-Api-Key": api_key}
    base_url = f"https://{api_host.strip().rstrip('/')}"

    try:
        # 第一步先把用户输入的城市名解析为经纬度；天气查询接口使用坐标而非城市文本。
        with httpx.Client(
            base_url=base_url,
            headers=headers,
            # 设置单次 HTTP 请求超时，避免外部服务无响应时一直卡住工具执行。
            timeout=10.0,
        ) as client:
            geo_response = client.get(
                "/geo/v2/city/lookup",
                params={"location": city, "range": "cn", "number": 1, "lang": "zh"},
            )
            geo_response.raise_for_status()
            geo_data = geo_response.json()

            locations = geo_data.get("location") or []
            if geo_data.get("code") != "200" or not locations:
                return f"没有找到城市：{city}"

            # 有多个同名候选时当前实现取第一个；更严格的产品流程可要求用户确认地区。
            location = locations[0]
            # 第二步使用地理查询返回的经纬度请求当前天气数据。
            weather_response = client.get(
                f"/weather/v1/current/{location['lat']}/{location['lon']}",
                params={"lang": "zh-hans", "localTime": "true"},
            )
            weather_response.raise_for_status()
            weather = weather_response.json()

    # 将常见网络错误转成工具可返回的文本，让 Agent 可以解释失败；当前不自动重试。
    except httpx.HTTPStatusError as exc:
        return f"天气接口返回 HTTP {exc.response.status_code}，请检查 API Host、凭据和接口权限。"
    except httpx.RequestError as exc:
        return f"天气接口请求失败：{type(exc).__name__}"

    # 从服务响应中挑选面向用户的字段，而不是把完整 JSON 原样塞回模型上下文。
    condition = weather.get("condition", {}).get("text", "未知")
    temperature = weather.get("temperature", {})
    feels_like = weather.get("feelsLike", {})
    humidity = weather.get("humidity")

    # 当前代码假设 humidity 是 0~1 比例并乘 100；需以当前接口 Schema/实际响应核对单位。
    # 若服务返回的已是 0~100 百分比，这里就不应再乘 100。
    humidity_text = (
        f"{round(humidity * 100)}%"
        if isinstance(humidity, (int, float))
        else "未知"
    )
    place_name = location.get("adm1", "") + location.get("name", city)
    # 优先使用服务响应提供的 attribution；响应未提供时给出默认来源名称。
    attributions = weather.get("metadata", {}).get("attributions") or []
    source = "；数据来源：" + "、".join(attributions) if attributions else "；数据来源：和风天气"

    return (
        f"{place_name}实时天气：{condition}，"
        f"气温 {temperature.get('value', '未知')}{temperature.get('unit', '')}，"
        f"体感 {feels_like.get('value', '未知')}{feels_like.get('unit', '')}，"
        f"湿度约 {humidity_text}{source}"
    )


if __name__ == "__main__":
    # stdio 是 MCP Client 与 Server 的协议通道；不要在此向 stdout 输出普通调试日志。
    mcp.run(transport="stdio")
