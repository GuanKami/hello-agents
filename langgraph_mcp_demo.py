"""实验名称：通过 MCP 接入真实天气服务

实验目标：把独立 MCP Server 暴露的天气能力接入 LangChain/LangGraph Agent。
解决的问题：让 Agent 不再依赖写在 Python 函数里的固定天气文本，而是通过外部天气服务查询实时数据。
使用的 Agent 概念：MCP、stdio 传输、工具发现、Tool Calling、Agent 工具循环。
系统架构：用户 -> create_agent -> MCP Client -> stdio 子进程 -> FastMCP Server
          -> 城市地理查询 -> 实时天气查询 -> MCP 工具结果 -> Agent -> 用户。
实现方式：MultiServerMCPClient 使用当前解释器启动 server.py，调用 get_tools() 发现工具，
          再把适配后的工具交给 create_agent；Agent 通过 ainvoke 异步运行。
验证方式：维护者曾验证 MCP 工具发现和工具调用循环；本文件对应的最新 QWeather 实时接口
          路径在本次修改中只做静态检查，没有重新发起天气或 LLM 请求。
学习总结：MCP Server 与 Agent 解耦；客户端通过协议发现能力，Agent 无需直接导入服务端函数。
已知限制：天气服务仅查中国城市；配置、网络、账户权限和模型是否选用工具都会影响结果；
          用户请求中的“今天”由当前天气接口回答，并不代表逐日天气预报。
后续优化方向：补充配置预检、明确的预测天气接口、结构化结果、超时/重试策略和可重复的本地测试。
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

# 模型配置从仓库根目录的 .env 读取；不要打印或提交密钥。
load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


async def main() -> None:
    # 以本文件所在位置定位 MCP Server，避免依赖 PowerShell 当前工作目录。
    project_root = Path(__file__).resolve().parent
    server_path = project_root / "mcp_server" / "get_weather_mcp" / "server.py"

    client = MultiServerMCPClient(
        {
            "weather": {
                # MCP stdio 模式会把 Server 作为子进程启动。
                # 使用当前解释器，确保它能导入当前环境安装的 fastmcp/httpx 等依赖。
                "command": sys.executable,
                "args": [str(server_path)],
                "transport": "stdio",
            }
        }
    )

    # 握手并向 Server 获取工具定义；返回值是适配后的 LangChain 工具，
    # Agent 可据此生成结构化 tool_call，MCP Client 再把调用转发给 Server。
    tools = await client.get_tools()

    print("发现的 MCP 工具：")
    print([tool.name for tool in tools])

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant. "
            "When the user asks about weather, use the weather tool."
        ),
    )

    # Agent 会根据用户问题决定是否调用 get_weather，并在收到 ToolMessage 后继续生成回答。
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请查询北京今天的天气。",
                }
            ]
        }
    )

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
