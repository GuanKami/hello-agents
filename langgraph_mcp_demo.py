"""实验名称：使用 MCPAdapter 接入真实天气 MCP Server

实验目标：把独立 FastMCP 天气服务作为 MCP 工具接入 `create_agent`。
解决的问题：Agent 不直接导入 Server 函数或耦合天气服务实现，而是通过 MCP 协议发现并调用外部能力。
使用的 Agent 概念：MCP Client/Server、stdio、工具发现、结构化 Tool Calling、ReAct 工具循环、异步资源生命周期。
系统架构：用户 -> create_agent -> MCPAdapter -> stdio 子进程 -> FastMCP Server
          -> 城市解析 / 当前天气 HTTP API -> MCP 工具结果 -> Agent -> 用户。
实现方式：使用 `langchain[mcp]` 内置的 `MCPAdapter`，在异步上下文中保持连接；
          `list_tools()` 发现工具后交给 `create_agent`，再通过 `ainvoke()` 执行 Agent。
验证方式：2026-09-26 的两次真实运行均发现 `get_weather`、成功查询北京并收到 ToolMessage，
          最终模型依据结果回答；目前只验证了两次单工具请求。
学习总结：MCPAdapter 负责连接和工具适配；Agent 负责决定是否调用工具及如何利用结果继续回答。
已知限制：天气服务仅查中国城市；配置、网络、服务权限和模型决策都会影响结果；
          此处调用当前天气接口，不是逐日天气预报；异常分支和其他城市尚未验证。
后续优化方向：改善地点名称格式，补充配置预检、预测天气接口和有边界的重试策略，
          再考虑无需真实模型/天气 API 的本地协议级测试。
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
# MCPAdapter 是 LangChain 内置的 MCP 集成入口（当前仍为 Beta）；本实验不再使用旧的
# langchain-mcp-adapters.MultiServerMCPClient。
from langchain.mcp import MCPAdapter
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

    # mcpServers 按名称描述 MCP Server。command + args 指定以 stdio 子进程方式启动；
    # sys.executable 保证子进程使用当前虚拟环境，从而能导入 FastMCP 和 httpx。
    # MCPAdapter 会根据 command/args 推断 stdio 传输，因此这里不再配置旧式 transport 字段。
    adapter_config = {
        "mcpServers": {
            "weather": {
                "command": sys.executable,
                "args": [str(server_path)],
            }
        }
    }

    # MCPAdapter 是异步资源：上下文管理器负责建立并在退出时关闭连接/子进程。
    # 必须把 Agent 的整个 ainvoke 放在上下文内部，因为模型可能在运行期间才调用 MCP 工具；
    # 若发现完工具就退出上下文，后续工具调用时连接已被关闭。
    async with MCPAdapter(adapter_config) as adapter:
        # list_tools() 与 MCP Server 协商并读取其工具定义，再转换为 LangChain 可执行工具。
        # create_agent 接收这些工具后，模型产出的 tool_call 会由 Agent 执行流程路由到对应 MCP Server。
        tools = await adapter.list_tools()

        print("发现的 MCP 工具：")
        print([tool.name for tool in tools])

        # create_agent 隐藏了 ReAct 的重复循环：模型若产生 tool_call，框架执行工具、
        # 把工具结果追加为 ToolMessage，再调用模型生成最终回答；无需手写循环。
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=(
                "You are a helpful assistant. "
                "When the user asks about weather, use the weather tool."
            ),
        )

        # ainvoke 异步运行完整 Agent 流程。真实调用会访问模型 API；若模型选择天气工具，
        # MCPAdapter 会把调用经 stdio 转发给 Server，Server 再请求外部天气 HTTP API。
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

        # 打印完整消息轨迹，便于区分模型的 AIMessage(tool_calls)、工具返回的 ToolMessage，
        # 以及读取工具结果后生成的最终 AIMessage。
        for message in result["messages"]:
            message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
