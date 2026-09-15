# hello-agents

个人 AI Agent 学习实验仓库。

这个仓库用于通过一组小型、可运行、可观察的 Python 实验，逐步理解现代 AI Agent 的核心机制，并为后续学习字节跳动 DeerFlow、开发独立 Agent Demo 和完成简历项目打基础。

> 当前状态：学习阶段，尚未进入最终项目的正式实现阶段。
>
> 截至 2026 年 9 月 15 日，已经完成 LangGraph 快速入门、状态图、Memory、Context Engineering，以及部分 Middleware 实验。

## 项目目标

长期目标是：

1. 系统学习现代 AI Agent 架构。
2. 理解 DeerFlow 的核心工程思想。
3. 能够独立完成 Agent Demo。
4. 最终完成一个具有实际价值、可以写入简历的 AI Agent 项目。

当前维护者具备后端开发和 CRUD 应用开发基础，但仍在逐步深入学习：

- LLM 应用开发
- Prompt Engineering
- Tool Calling
- Agent Loop
- Planning / Reasoning
- Memory 与 Context Engineering
- RAG
- Workflow 编排
- Multi-Agent
- Agent Evaluation 与 Observability

## 当前学习主线

```text
LLM API
  -> Tool Calling
  -> LangChain Agent API
  -> LangGraph ReAct
  -> StateGraph 与工作流
  -> Memory / Context Engineering
  -> Middleware / Human-in-the-loop
  -> MCP / RAG / Subgraph / Multi-Agent
  -> Evaluation / Observability / Production
  -> DeerFlow 源码与架构学习
  -> RepoResearcher 简历项目
```

## 已完成内容

### Agent 基础

- LLM API 和 OpenAI 兼容接口调用。
- ReAct、Plan-and-Solve、Reflection 的基本思想和 Demo 阅读。
- LangChain `@tool`、工具 Schema 和结构化 Tool Calling。
- `ToolRuntime`、运行时 `Context` 和工具权限控制。
- `create_agent` 高层 Agent API。

### LangGraph 基础

- `StateGraph`、State、Node、Edge 和条件路由。
- `ToolNode` 执行结构化工具调用。
- 显式 StateGraph 版 ReAct：`model -> tools -> model`。
- `invoke`、`stream` 和流式输出观察。
- Mermaid 静态图与真实运行轨迹的区别。

### Memory 与 Context Engineering

- 短期记忆、checkpoint 和 `thread_id`。
- 长期记忆、`user_id`、Store 和结构化用户资料读写。
- `save_user_info` 的增量字典合并。
- Embedding 与 `InMemoryStore` 语义检索。
- `InMemoryStore` 的进程生命周期限制。
- `SqliteStore` 持久化机制的课程 Notebook 学习。
- Runtime + Store -> Model Context。
- State -> Model Context。
- `Context`、`Runtime`、`State`、`Store` 和 Model Context 的职责边界。

### Middleware

- `@dynamic_prompt`。
- `@before_model` / `@after_model` 作为图节点读写 State。
- 使用 `request.state` 观察 State。
- 使用 `request.override(model=...)` 动态切换模型。
- 使用 `wrap_model_call` 观察模型请求与响应。
- 在 ReAct 工具循环中观察 wrapper 的重复执行。
- 本地短路：命中条件时直接返回 `ModelResponse`，不继续调用内层 `handler`。

Middleware 的重试、`wrap_tool_call`、Human-in-the-loop、MCP、RAG、Subgraph、并行和 Multi-Agent 仍未完成，具体状态以 [AGENTS.md](AGENTS.md) 为准。

## 实验文件

| 文件 | 学习内容 | 当前状态 |
| --- | --- | --- |
| [`tools.py`](tools.py) | 共享工具、权限控制、用户资料读写 | 已使用 |
| [`langgraph_react.py`](langgraph_react.py) | 高层 `create_agent` Agent API | 学习实验，存在待修正缺陷 |
| [`langgraph_state_react.py`](langgraph_state_react.py) | 显式 StateGraph ReAct 和 ToolNode | 已完成基础实验 |
| [`embedding_test.py`](embedding_test.py) | Embedding、Store 和语义检索 | 已完成最小实验 |
| [`langgraph_context_demo.py`](langgraph_context_demo.py) | Runtime + Store 构造 Model Context | 已完成 |
| [`langgraph_state_context_demo.py`](langgraph_state_context_demo.py) | State 驱动 Model Context | 已完成 |
| [`langgraph_middleware_dynamic_prompt_demo.py`](langgraph_middleware_dynamic_prompt_demo.py) | `@dynamic_prompt` | 已完成 |
| [`langgraph_middleware_hooks_demo.py`](langgraph_middleware_hooks_demo.py) | `before_model` / `after_model` 写入 State，并复现动态换模型 | 已完成真实模型验证 |
| [`langgraph_middleware_wrap_model_call_demo.py`](langgraph_middleware_wrap_model_call_demo.py) | wrapper 观察、模型调用链和本地短路 | 进行中 |

根目录的 `dive-into-langgraph/` 是独立的课程源码仓库，必须保留；`.agents/skills/dive-into-langgraph/` 是本地学习 Skill，也必须保留。

## 目录和职责

当前项目保持平铺式学习脚本结构：

```text
hello-agents/
├── tools.py                                  # 共享工具
├── langgraph_react.py                         # 高层 Agent API 实验
├── langgraph_state_react.py                   # 显式 StateGraph ReAct
├── embedding_test.py                          # Embedding / Store 实验
├── langgraph_context_demo.py                 # Runtime + Store 上下文
├── langgraph_state_context_demo.py           # State 上下文
├── langgraph_middleware_dynamic_prompt_demo.py
├── langgraph_middleware_hooks_demo.py
├── langgraph_middleware_wrap_model_call_demo.py
├── requirements.txt
├── AGENTS.md
├── .agents/                                   # 本地 Skill 配置
└── dive-into-langgraph/                       # 课程资料，独立仓库
```

当前根目录没有正式测试目录、CI 配置、构建配置、部署配置或统一应用入口。每个脚本主要服务于一个学习目标，不应仅为了减少文件数量而过早抽象。

## 技术栈

- Python
- LangChain
- LangChain OpenAI
- LangGraph
- Pydantic
- python-dotenv
- OpenAI 兼容接口
- SerpApi（共享搜索工具使用）
- IPython（消息展示和 Notebook 支持）

根目录 `requirements.txt` 是当前明确声明的依赖列表。`langgraph` 和 `pydantic` 已被代码直接使用，但目前还没有在该文件中作为直接依赖单独声明；这属于后续工程整理事项。

## 环境配置

本地实验从根目录 `.env` 读取配置。`.env` 已被 `.gitignore` 排除，不能提交或打印其中的值。

当前代码使用过的配置名称包括：

```text
LLM_MODEL_ID
LLM_API_KEY
LLM_BASE_URL
EMBEDDING_MODEL_ID
MODEL_PROVIDER
SERPAPI_API_KEY
BASIC_MODEL_ID
ADVANCED_MODEL_ID
```

具体模型、接口地址和密钥由本地环境决定，README 不记录真实配置。

## 安装和运行

仓库目前没有官方安装脚本、启动脚本、测试脚本或构建脚本。已有依赖可根据 `requirements.txt` 安装，具体实验使用虚拟环境中的 Python 逐个运行。

在 Windows PowerShell 中，当前常用入口如下：

```powershell
.\.venv\Scripts\python.exe langgraph_state_react.py
.\.venv\Scripts\python.exe embedding_test.py
.\.venv\Scripts\python.exe langgraph_context_demo.py
.\.venv\Scripts\python.exe langgraph_state_context_demo.py
.\.venv\Scripts\python.exe langgraph_middleware_dynamic_prompt_demo.py
.\.venv\Scripts\python.exe langgraph_middleware_hooks_demo.py
.\.venv\Scripts\python.exe langgraph_middleware_wrap_model_call_demo.py
```

语法检查可以使用：

```powershell
.\.venv\Scripts\python.exe -m py_compile tools.py langgraph_react.py langgraph_state_react.py embedding_test.py
```

这些实验中的模型和 Embedding 调用会访问外部 API，可能产生费用。没有测试框架时，脚本运行结果属于手动验证，不能等同于自动化测试。

## 已确定的未来项目方向：RepoResearcher

最终项目方向已经确定为：

> RepoResearcher：面向代码仓库的研究与研发辅助 Agent。

这个项目目前只是规划，尚未进入实现阶段。它将来希望解决的问题是：帮助后端开发者快速理解陌生代码仓库、定位功能入口、追踪调用链、分析错误日志、执行受控验证，并生成带源码证据的技术报告。

第一版目标可以限定为：

```text
输入：本地 Python 仓库路径 + 一个研发问题
  ↓
搜索文件和代码
  ↓
读取关键文件
  ↓
整理证据和调用关系
  ↓
生成 Markdown 报告和 Mermaid 图
```

计划逐步加入的能力：

1. 单 Agent + 本地文件工具。
2. StateGraph 管理分析流程。
3. Checkpoint 支持任务恢复。
4. Context Engineering 管理当前问题、项目身份和证据。
5. RAG 检索项目文档和技术决策记录。
6. Human-in-the-loop 控制测试执行和代码修改。
7. Subgraph / Parallel 拆分代码探索、验证和报告生成。
8. MCP 扩展 GitHub、搜索和外部研发工具。
9. Evaluation / Observability 统计准确率、耗时、Token、失败原因和工具调用情况。
10. 对照 DeerFlow 的 Skills、Subagents、Memory、Sandbox、Context 和 Artifact 思路继续演进。

在正式实现 RepoResearcher 之前，先完成相关 LangGraph 章节和小实验。当前学习仓库中的实验文件会作为机制基线保留，不会直接被改造成最终项目代码。

## 后续学习路线

```text
Middleware
  -> wrap_model_call 重试 / fallback
  -> wrap_tool_call
  -> Human-in-the-loop
  -> MCP Server
  -> RAG
  -> Subgraph / Parallel / Map-Reduce
  -> Supervisor / Multi-Agent
  -> Evaluation / Observability
  -> RepoResearcher MVP
  -> DeerFlow 源码对照与工程化演进
```

每次实验只引入一个主要新概念，并保留可观察的验收点，例如：

- 模型实际生成了哪些工具调用。
- 工具节点如何执行并返回 `ToolMessage`。
- State 在节点之间如何变化。
- 不同用户和会话如何隔离。
- wrapper 是否调用了下游 `handler`。
- 中断后是否可以恢复。
- RAG 返回的证据是否支持最终结论。

## 开发原则

- 先理解机制，再扩展功能。
- 保持实验小而清晰，避免过早引入复杂抽象。
- 新增工具必须有准确的类型注解和 docstring。
- 明确区分 State、Context、Store、Runtime 和 Model Context。
- 结构化事实优先使用精确 key 查询，语义资料再使用向量检索。
- 高风险工具需要权限控制和人工确认。
- 不把模型生成的代码直接交给宿主进程执行。
- 外部 API、费用、持久化和安全边界必须在代码注释或实验文档中说明。
- 重要实验需要记录目标、架构、实现方式、验证方式、已知限制和后续方向。

## 相关资料

- [`AGENTS.md`](AGENTS.md)：本仓库的长期协作规范、真实进度和待办。
- [`dive-into-langgraph/`](dive-into-langgraph/)：本地课程源码和 Notebook。
- [Dive Into LangGraph 在线资料](https://luochang212.github.io/dive-into-langgraph/)
- [LangChain 官方文档](https://docs.langchain.com/oss/python/langchain/overview)
- [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/overview)
- [DeerFlow](https://github.com/bytedance/deer-flow)：长期学习目标，不是当前仓库的运行时依赖。

## 许可和状态说明

这是个人学习实验仓库，当前不承诺生产可用性。实验代码可能包含教学简化、固定演示数据、未完善的错误处理和需要外部模型服务的部分。随着学习推进，实验会逐步重构为更接近工程实践的实现。
