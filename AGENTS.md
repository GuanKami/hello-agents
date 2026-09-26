# AGENTS.md

本文件是 `hello-agents` 仓库的长期协作规范，面向未来协助本仓库的 AI Coding Agent，也面向仓库维护者本人。仓库当前是个人学习实验仓库，不是已经完成的生产系统。任何新增代码都应服务于学习目标，并逐步提高工程质量。

## 1. 项目背景（Project Context）

这是一个用于学习现代 AI Agent 开发的 Python 实验仓库。当前代码以平铺式脚本为主，目标是通过可运行、可观察的小实验理解 Agent 核心机制，逐步成长为能够设计和实现复杂 Agent 系统的工程师。

维护者的长期目标是：

1. 系统学习现代 AI Agent 架构。
2. 理解字节跳动 DeerFlow 的核心工程思想。
3. 能够独立完成 Agent Demo。
4. 最终完成一个具有实际价值、可以写入简历的 AI Agent 项目。

当前学习主线为：

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
```

根目录的 `dive-into-langgraph/` 是课程源码、Notebook、示例和参考资料目录，必须保留，不属于待清理的业务代码。`.agents/skills/dive-into-langgraph/` 是本仓库配置的本地学习 Skill，也必须保留。

DeerFlow 是本仓库的长期学习目标，不是当前仓库的运行时依赖。阅读 DeerFlow 时，必须以实际源码、文档和当前版本行为为依据，不要把本仓库的实验代码描述成 DeerFlow 的实现。

## 2. 开发者背景（Developer Context）

仓库维护者具备：

- 后端开发基础。
- CRUD 类型应用的独立开发能力。
- 基础软件工程和代码阅读能力。
- 对 Python、LLM 应用和 Agent 工程仍处于逐步深入阶段。

未来 AI Agent 在协助开发时必须：

- 不要默认维护者已经理解高级 Agent 概念。
- 解释关键设计思想，不只给出最终代码。
- 尽量把 State、Node、Edge、Checkpoint、Tool 映射到业务状态、Service、流程转移、可恢复快照和受控外部接口。
- 说明代码解决了什么问题、为什么这样设计、当前有什么限制。
- 学习性修改默认优先给出目标、改动点、代码骨架和验证方法，让维护者自己完成实现；只有明确要求代为修改时才直接编辑学习代码。
- 使用高层 API 时，说明它隐藏了哪些底层 LangGraph 机制。

## 3. 当前仓库事实和目录结构

### 3.1 根目录

当前根目录主要内容如下：

```text
.
├── AGENTS.md
├── tools.py
├── langgraph_react.py                                  # 高层 Agent API 实验
├── langgraph_state_react.py                            # 显式 StateGraph ReAct
├── embedding_test.py                                   # Embedding / Store 语义检索
├── langgraph_context_demo.py                           # Runtime + Store -> Model Context
├── langgraph_state_context_demo.py                     # State -> Model Context
├── langgraph_middleware_dynamic_prompt_demo.py         # @dynamic_prompt
├── langgraph_middleware_hooks_demo.py                  # before_model / after_model（已验证）
├── langgraph_middleware_wrap_model_call_demo.py        # wrap_model_call（短路已观察，重试未做）
├── langgraph_middleware_wrap_tool_call_demo.py         # wrap_tool_call（工具观察实验，已验证）
├── langgraph_middleware_tool_guard_demo.py             # 工具权限校验与执行短路（已验证）
├── langgraph_middleware_tool_error_demo.py             # 工具异常转换实验（ZeroDivisionError 已验证）
├── langgraph_human_in_the_loop_demo.py                 # Human-in-the-loop 审批（approve/reject 已验证）
├── langgraph_human_in_the_loop_sqlite_demo.py          # SqliteSaver 跨进程 HITL（approve/reject 已验证）
├── langgraph_mcp_demo.py                               # MCPAdapter + Agent，接入真实天气 MCP
├── mcp_server/
│   └── get_weather_mcp/server.py                       # FastMCP 天气服务端，调用和风天气接口
├── .gitignore            # 版本控制排除规则，见 3.4 节
├── requirements.txt
├── short-memory.db       # 本地 SQLite 运行产物；当前无脚本写入，见 4.11 节
├── hitl-checkpoint.db     # HITL SQLite checkpoint 运行产物；由 4.17 节脚本生成
├── skills-lock.json      # 本地 Skill 相关锁定信息
├── .env                  # 本地敏感配置，不应提交或打印
├── .agents/              # 本地 Agent/Skill 配置
├── .claude/              # 本地 Claude Code 权限配置（settings.local.json）
├── dive-into-langgraph/  # 课程源码和学习资料，必须保留
├── .venv/                # 本地虚拟环境
├── __pycache__/          # Python 运行产物
├── .idea/                # IDE 配置
└── .omo/                 # 本地工具配置或运行产物
```

当前根目录没有正式测试目录、CI 配置、构建配置、部署配置或统一应用入口。README.md 记录仓库定位、实验目录、运行方式和已确定但尚未实现的 RepoResearcher 方向。不要凭空假设这些设施存在。

### 3.2 本次清理记录

2026 年 9 月 7 日，按维护者明确要求，删除了以下已经不再服务于当前 LangGraph 学习主线的根目录旧脚本：

```text
ReAct.py
Plan_and_solve.py
Reflection.py
llm_client.py
```

删除原因：

- 它们属于早期手写 Agent 实验。
- `ReAct.py` 和 `Plan_and_solve.py` 依赖当前 `tools.py` 中已经不存在的旧 `ToolExecutor`。
- `Reflection.py` 依赖旧的 `llm_client.py`，并直接使用 `exec()` 执行模型生成代码。
- 当前主线已经转向结构化 Tool Calling、StateGraph、Memory 和 Context Engineering。

早期 ReAct、Plan-and-Solve 和 Reflection 的概念仍可作为学习背景讨论，但根目录已经没有对应源码文件，不得把它们描述成当前可运行入口。

`dive-into-langgraph/` 和 `.agents/skills/dive-into-langgraph/` 不属于清理目标，任何常规整理都不得删除、移动或覆盖它们。

### 3.3 文档同步纪律（重要）

`AGENTS.md` 曾经出现严重滞后：2026 年 9 月 8 日至 10 日新增的 4 个 Context Engineering 与 Middleware 实验文件，在文档中完全缺失，导致后续接手的 Agent 误判学习进度。

因此确立以下规则：

- **新增、重命名或删除根目录实验文件时，必须同步更新 3.1 目录结构。**
- **某个实验完成或状态变化后，必须同步更新 5.1 已完成清单、5.2 当前阶段和第 12 章 TODO。**
- **实际执行顺序偏离 5.3 节计划时，必须在 5.2 节如实记录偏离事实**，不得按原计划口径描述未完成的工作。
- **文档必须以当前代码和可验证结果为依据。** 无法验证的能力要写明“未验证”，不得根据文件名或过往描述推断。
- 修改代码后未同步文档，视为任务未完成。

### 3.4 版本控制现状

截至 2026 年 9 月 10 日：

```text
根目录 hello-agents/      -> 已建立 git 仓库（2026-09-10），有 .gitignore
dive-into-langgraph/      -> 是独立的 git 仓库（课程源码），与根项目实验代码无关
```

已建立的提交：

```text
a498db3  chore: 建立版本控制基线       <- 12 个源码文件的初始快照，可回滚到此
5176f14  feat(middleware): before_model/after_model 真正写入 Agent State
49cce20  docs(agents): 同步 before_model/after_model 实验完成状态与 git 安全网
```

注意：`5176f14` 提交的 `langgraph_middleware_hooks_demo.py` 是一个**已被维护者废弃的版本**（含假模型自检路径）。维护者随后按自己的版本重写了该文件，当前工作区版本才是有效版本。**不要用 `git checkout` 把它恢复到 `5176f14`。**

`.gitignore` 排除项（已用 `git check-ignore` 逐条验证生效）：

```text
.env                          <- 含真实密钥，绝不提交
.venv/  __pycache__/          <- 环境与运行产物
*.db  (含 short-memory.db)    <- 本地数据库
.idea/  .omo/                 <- 本地工具与 IDE 配置
.claude/settings.local.json   <- 含本机绝对路径的本地权限配置
dive-into-langgraph/          <- 嵌套的课程仓库，不纳入根仓库跟踪
```

注意事项：

- 根目录现在有版本历史，**改动实验代码前后可用 `git diff` / `git checkout` 对照与回滚**。
- **不要**在根目录用 git 命令操作 `dive-into-langgraph/`，它们是两个独立仓库。
- `dive-into-langgraph/` 内已存在的未提交修改属于课程内容，除非维护者明确要求，不要提交或还原。

## 4. 技术栈、配置和实验代码

### 4.1 技术栈和依赖

项目主要使用：

- Python。
- `langchain[mcp]`：Agent API、消息和工具抽象，并提供内置 `MCPAdapter`（当前 Beta）。
- `langchain-openai`：通过 OpenAI 兼容接口创建 `ChatOpenAI` 和 `OpenAIEmbeddings`。
- `langgraph`：StateGraph、ToolNode、checkpoint 和 Store 运行时能力。
- `pydantic`：`Context` 等运行时数据结构。
- `python-dotenv`：加载 `.env`。
- `google-search-results`：为 `tools.py` 提供 SerpApi 客户端。
- `openai`：OpenAI 兼容 SDK 的底层依赖。
- `ipython`：消息展示和 Notebook 支持。
- `fastmcp`：实现独立 MCP Server。
- `httpx`：天气 MCP Server 调用外部天气 HTTP API。
- 和风天气服务：运行时外部 API，不是 Python 包；需配置服务 Host 和密钥。

`requirements.txt` 当前明确声明的主要运行依赖包括：

```text
openai>=2.0
python-dotenv>=1.2.2
google-search-results>=2.4.2
ipython>=9.0
pydantic>=2.13
langchain>=1.3.11
langchain[mcp]>=1.4.2
langchain-openai>=1.3.3
langgraph>=1.2.6
langgraph-checkpoint-sqlite>=3.1.0
fastmcp>=4.0.10,<5
httpx>=0.27
```

截至 **2026 年 9 月 26 日**，本地环境实际观察到 Python 3.13.1、LangChain 1.4.2、LangChain OpenAI 1.4.1、LangGraph 1.2.12、FastMCP 4.0.10。这些是本机环境事实，不是项目锁定版本。


### 4.2 环境变量

`.env` 中使用过以下配置名称，文档、日志和回答中不得输出其值：

- `LLM_MODEL_ID`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `EMBEDDING_MODEL_ID`
- `MODEL_PROVIDER`
- `SERPAPI_API_KEY`
- `BASIC_MODEL_ID`（`langgraph_middleware_hooks_demo.py` 的低费率模型，2026-09-10 新增）
- `ADVANCED_MODEL_ID`（同上，高费率 / 默认模型）
- `QWEATHER_API_HOST`（天气 MCP Server 使用的天气 API Host；只记录名称，不记录真实值）
- `QWEATHER_API_KEY`（天气服务凭据；不得打印或提交）

仓库已有根目录 `.gitignore`（2026-09-10 新增，内容与排除项见 3.4 节）。新增敏感配置或运行产物目录时，必须同步补充排除规则，并用 `git check-ignore -v <路径>` 验证生效。

### 4.3 `tools.py`

`tools.py` 是当前共享工具模块，包含：

- `search(query)`：通过 SerpApi 搜索互联网，并返回答案框、知识图谱或有机搜索摘要。
- `calculate(runtime, a, b)`：通过 `ToolRuntime[Context, Any]` 读取权限，仅允许 `admin` 使用加法工具。
- `get_weather(city)`：固定返回演示天气文本，不是真实天气服务。
- `get_user_info(runtime)`：从 `runtime.store` 按 `("users",)` 和当前 `user_id` 精确读取用户资料。
- `save_user_info(user_info, runtime)`：读取旧资料，将新资料合并后写回 Store；相同 key 由新值覆盖旧值。

注意：`tools.py` 中的 `get_weather` 仍是固定文本的教学工具；真实天气接入由独立的
`mcp_server/get_weather_mcp/server.py` 提供，两者不是同一个实现。

当前 `Context` 和 `UserInfo` 的主要结构为：

```python
class Context(BaseModel):
    authority: Literal["admin", "user"]
    user_id: str


class UserInfo(TypedDict, total=False):
    name: str
    language: str
    favorite_topics: list[str]
    birth_year: int
    sex: Literal["male", "female", "other"]
    identity: str
```

`identity` 是后期新增的字段（2026 年 9 月 9 日左右加入，早于本文件首次记录），用于承载“从业者 / 厨师”这类身份描述。4 个 context 与 middleware 实验文件都在读写它。文档曾长期漏记该字段，属已修正的漂移。

工具规则：

- 工具使用 `@tool`、准确的类型注解和 docstring。
- `user_id` 和权限应从运行时上下文取得，不应由模型自行伪造或任意指定。
- 结构化用户资料使用精确的 namespace/key 查询，不需要 Embedding。
- 语义长期记忆才使用 `store.search()` 和 Embedding。
- 资料更新采用“读旧值、合并、写回”模式；当前实现尚未处理并发更新冲突。

### 4.4 `embedding_test.py`

这是独立的 Embedding 和 Store 语义检索实验，已验证文本可通过 OpenAI 兼容 Embedding 接口转换为**当前实测 4096 维**向量，并由 `InMemoryStore` 建立语义索引、执行精确 `get()` 与语义 `search()`。

该文件还使用 Agent 通过 `get_user_info` 和 `save_user_info` 读写结构化用户资料。它是 Memory 学习成果，也是未来语义长期记忆和 RAG 实验的基础，因此保留。

**文件头部注释质量最高，应作为后续实验文档的参照样板**：`实验名称 / 实验目标 / 解决的问题 / 使用的 Agent 概念 / 系统架构 / 实现方式 / 验证方式 / 学习总结 / 已知限制 / 后续优化方向` 十项齐全，且记录了真实的失败教训（供应商切换、账户级限额、向量空间不通用）。

**已知的内部矛盾（维度漂移，截至 2026 年 9 月 10 日尚未修正）**：同一文件内出现三个互相冲突的维度数字：

```text
第 12 行  "文本 --[embeddings]--> 1024 维向量"     <- 旧注释，错误
第 19 行  "输出维度是否恒为 1024"                   <- 旧注释，错误
第 81 行  "本模型实测 2048"                         <- 旧注释，错误
第 90 行  EMBED_DIM = 4096                          <- 代码，正确
```

同一份文件头部第 9 行又写"qwen/qwen3-embedding-8b（4096 维）"。因此**唯一可信来源是第 90 行的 `EMBED_DIM = 4096` 与实际运行打印的维度**。阅读时不要被 1024 / 2048 误导。

其他已知限制：

- `InMemoryStore` 只在当前 Python 进程保存数据，退出即丢失。
- 默认索引会嵌入整个 value 的 `str()` 形式（含 dict 键名），属结构噪声，会稀释语义信号；语义记忆应改为自然语言事实句并用 `fields=["text"]` 限定索引字段。
- 文件头声明的"硅基流动方案已弃用"等历史结论反映的是过去某一时刻的供应商状态，`.env` 实际配置可能已经变化，不要当作当前事实。
- 保留了从 ReAct 实验复制过来、本文件用不到的 import（如 `sqlite3`、`ToolNode`、`SqliteSaver`、`InMemorySaver`）。

### 4.5 `langgraph_context_demo.py`：Runtime + Store → Model Context

演示"长期记忆与运行时身份如何变成模型上下文"：

```text
runtime.context.user_id   -> 本次运行的真实身份（应用代码注入，模型无法伪造）
runtime.store.get(...)    -> 按 user_id 精确取出该用户长期资料
字段裁剪                  -> 只挑选当前任务需要的字段，不整份塞给模型
SystemMessage             -> 注入本次模型调用
```

关键实现事实：

- 图由 `StateGraph(MessagesState, context_schema=Context)` 编译，`compile(store=store)` 注入 Store。
- `model_node(state, runtime: Runtime[Context])` 内读取 `runtime.context.user_id` 与 `runtime.store`。
- 动态构造的 `SystemMessage` **只拼进本次调用的 messages 列表，不写回 `state["messages"]`**。若写回，ReAct 循环每轮都会再追加一条 system 消息，造成上下文累积污染。
- 验收点：同一问题在 `user_1`（派大星/中文）与 `user_2`（猪八戒/English）两个 `thread_id` 下得到不同语言与资料的回答。

### 4.6 `langgraph_state_context_demo.py`：State → Model Context

演示"工作流状态如何变成模型上下文"，与 4.5 形成来源对照：

```python
class AgentState(MessagesState):
    task_mode: NotRequired[Literal["learning", "general"]]
```

- 新增真实图节点 `classify_task`，按关键词判定任务类型并写回 State；边为 `START -> classify_task -> model`。
- `model_node` 读取 `state.get("task_mode", "general")`，查表得到 `task_instruction`，与用户资料、语言规则一起拼进同一个 `SystemMessage`。
- 学习要点：4.5 的上下文来源是 **Context + Store**（请求级 / 长期），本实验的来源是 **State**（工作流级，可被 checkpoint）。两者最终都产出 SystemMessage，但语义与生命周期不同。
- 已知教学简化：`classify_task` 使用硬编码关键词列表，不是模型分类。生产实现通常用模型路由或分类模型。

### 4.7 `langgraph_middleware_dynamic_prompt_demo.py`：`@dynamic_prompt`

第一个 Middleware 实验，切换到高层 `create_agent`：

```python
@dynamic_prompt
def user_aware_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id
    item = request.runtime.store.get(("users",), user_id)
    return "...动态系统提示词..."
```

**必须记住的机制事实**：`@dynamic_prompt` 不是独立 hook，它是 `wrap_model_call` 的便捷封装。已安装的 langchain 1.3.14 源码中该装饰器文档字符串明确写道：

> "This is a convenience decorator that creates middleware using `wrap_model_call` specifically for dynamic prompt generation."

其实现本质为：

```python
def wrapped(_self, request, handler):
    prompt = func(request)
    request = request.override(system_message=SystemMessage(content=prompt))
    return handler(request)
```

因此它生成的是 `wrap_model_call` / `awrap_model_call`，**不产生任何图节点，也不写入 State**。这与 `before_model` / `after_model`（真正的图节点，可写 State）是本质区别，是后续 Middleware 学习的关键分界。

### 4.8 `langgraph_middleware_hooks_demo.py`：`before_model` / `after_model`

当前状态：**核心机制已由真实模型运行实测通过（2026-09-10）。两个学习目标均已验证。**

文件结构：

```text
文件头十项 docstring（按 6.4.3 规范）
basic_model / advanced_model           <- 由 BASIC_MODEL_ID / ADVANCED_MODEL_ID 构造
CallCountState(AgentState)             <- 扩展 model_call_count / last_tool_names
build_store()                          <- 依赖注入，便于将来换 SqliteStore
count_model_calls    (@before_model)   <- 返回 {"model_call_count": n}
inspect_model_output (@after_model)    <- 返回 {"last_tool_names": [...]}
dynamic_model_selection(@wrap_model_call) <- 复现课程"预算控制"，按消息数换模型
build_agent(store)                     <- 组装 create_agent（三个 middleware 全注册）
print_messages() / print_state_summary()
main()                                 <- 真实模型运行，有 API 成本
```

**该文件现在承载两个概念**（2026-09-10 维护者加入第二个）：

- 主概念：`before_model` / `after_model` 写 State。
- 附带：复现课程 `3.middleware.ipynb` 第一章的 `@wrap_model_call`"预算控制"——读 `request.state["messages"]` 长度，超过阈值就 `request.override(model=basic_model)`。按 6.1"每个实验只引入一个主要新概念"，附带部分在文件头已明确标注"不是本实验的学习目标"。**文件名的语义已略窄于文件内容**，是否拆分由维护者决定。

**已实测确认的层叠关系**：`wrap_model_call` 包裹器位于**模型节点内部**，图节点在外。运行时打印顺序实证为：

```text
[before_model] 第 1 次调用模型, 当前 State 消息数 = 5     <- 图节点，先执行
message_count: 5                                          <- 包裹器，后执行
model_name: nvidia/nemotron-3.5-lightning:free
[after_model] 模型产出 tool_calls = ['get_user_info']     <- 图节点
```

由此得到一个**反直觉但重要**的推论：包裹器短路（不调用 `handler`）时，`before_model` **早已执行完毕**，所以 **`before_model` 的计数不能用作"模型是否真的被调用"的探针**。正确探针是响应自身的元数据——真实响应带 `response_metadata.model_name` / `usage_metadata`，而本地伪造的 `AIMessage(content=...)` 这两项为空（已做零成本本地实测：`response_metadata = {}`、`usage_metadata = None`、`id = None`）。

**课程"预算控制"复现的验证结果**：`message_count = 1` 时用 `advanced_model`；`message_count = 5` 时切到 `basic_model`，与代码阈值 `> 4` 一致。一次验证三件事：

1. `wrap_model_call` 里能**读** `request.state`（只读，**不能写**）。
2. `request.override(model=...)` 能替换本次调用的模型。
3. `handler(request)` 才真正发起调用；不调用 `handler` 就没有请求。

**尚未覆盖**：`handler` 被调用 **0 次**（短路）与 **N 次**（重试）。

说明：本实验**刻意只用真实模型验证，不引入假模型 / Mock**（维护者明确要求）。因此验证不是确定性的，每次运行都会产生真实的 LLM API 调用与费用，且结果受模型当时行为影响。文件内没有 `run_deterministic_selfcheck()` 之类的自检路径。

**已实测通过的验证（真实模型，符合预期）**：

```text
请求 1（"你好"，不触发工具）：
[before_model] 第 1 次调用模型, 当前 State 消息数 = 1
[after_model]  模型产出 tool_calls = []
最终 model_call_count = 1，last_tool_names = []

请求 2（"北京今天天气怎么样？"，触发工具）：
[before_model] 第 1 次调用模型, 当前 State 消息数 = 1
[after_model]  模型产出 tool_calls = ['get_weather']
[before_model] 第 2 次调用模型, 当前 State 消息数 = 3
[after_model]  模型产出 tool_calls = []
最终 model_call_count = 2，last_tool_names = []
```

两个学习目标的验证结果：

1. **触发次数 = 模型调用次数**：不触发工具时 hook 各 1 次；触发工具时各 2 次。
2. **返回 dict 确实写入 State**：`model_call_count` 由 `result.get(...)` 从 `invoke` 的**返回值**中读出（请求 1 得 1，请求 2 得 2），证明 patch 被合并进 State 且 invoke 返回后依然可读。这是与 `@dynamic_prompt`（不写 State）的分界线。

两处 `last_tool_names = []` 属预期，不是"工具未被调用"：该字段是**标量字段，reducer 为覆盖**，第 1 轮写入 `['get_weather']` 后被第 2 轮的 `[]` 覆盖。因此它表示"**最后一次**模型调用请求了哪些工具"，不是本次运行的累计工具使用记录。

**已实测确认的图结构**（`agent.get_graph()` 的节点与边）：

```text
节点: model, tools, count_model_calls.before_model, inspect_model_output.after_model
边:   START -> before_model -> model -> after_model
      after_model --条件--> END
      after_model --条件--> tools
      after_model --条件--> before_model
      tools --条件--> before_model
```

由此可见 `tools -> before_model -> model -> after_model` 构成循环，这正是"模型调用几次，两个 hook 就各触发几次"的源码级原因。

关键源码事实：

- hook 被编译为真正的图节点：`graph.add_node(f"{m.name}.before_model", ...)`。
- 循环入口为第一个 `before_model`：`loop_entry_node = f"{middleware_w_before_model[0].name}.before_model"`。
- 多个 `after_model` **逆序**串联（`range(len-1, 0, -1)`），因为它是栈式包裹。
- `state_schema=` 是装饰器的关键字参数（`before_model(func=None, *, state_schema=None, tools=None, can_jump_to=None, name=None)`）。它只做声明，不做类型检查。

**必须记住的坑**：hook 返回的自定义 State 字段**必须**出现在合并后的 State schema 中，否则会被**静默丢弃且不报错**。已实测：未声明 `state_schema` 时返回 `{"model_call_count": 99}`，最终读出 `None`；声明后读出 `99`。

**本文件未覆盖的部分**：多个同类型 middleware 的组合顺序；`can_jump_to` 跳转；消息裁剪；`wrap_model_call` 的短路 / 重试（短路已在 4.12 节独立实验观察，重试仍未做）；`wrap_tool_call`。本实验刻意不使用假模型 / Mock，因此不存在"不联网、可重复"的确定性验证路径——每次验证都是一次真实模型运行。

### 4.9 `langgraph_react.py`

这是高层 Agent API 实验：

```text
create_agent
  -> 模型决定是否调用工具
  -> Agent 内部执行工具循环
  -> 返回 HumanMessage / AIMessage / ToolMessage
```

它用于对比高层 `create_agent` 和显式 StateGraph 的差异。当前文件使用 `InMemorySaver` 和 `thread_id` 演示短期会话。

**已知缺陷（截至 2026 年 9 月 10 日仍未修复）**：第 38 行和第 47 行的 `Context(authority="admin")` 没有提供必填的 `user_id`。已实测该文件**在 import 阶段就会抛出 `pydantic ValidationError`**，因为文件顶层的 `tool_agent.invoke(...)` 在模块加载时执行。任何 `import langgraph_react` 都会失败。修正前不要把它描述成可运行入口；文件还包含未使用导入。

### 4.10 `langgraph_state_react.py`

这是显式 StateGraph 版本的 ReAct 和长期资料实验：

```text
START
  -> model
  -> 条件路由
     ├── 有 tool_calls -> tools -> model
     └── 无 tool_calls -> END
```

当前明确展示：

- `MessagesState` 保存消息状态。
- 模型节点调用绑定工具的模型。
- `ToolNode` 执行结构化工具调用。
- 条件边决定继续执行工具还是结束。
- `tools -> model` 的循环形成 ReAct 行为。
- `store=store` 将 Store 注入 `ToolRuntime`。
- `graph.get_graph().draw_mermaid()` 只输出静态图描述；真实执行仍由 `invoke` 或 `stream` 完成。

当前 `SqliteSaver` 的导入、实例化和 `compile(checkpointer=...)` 已注释，用于隔离长期 Store 实验。当前仍使用 `InMemoryStore`，所以资料只在同一个 Python 进程中有效。

**注意 import 副作用**：该文件在第 36–42 行于**模块加载阶段**调用 `embeddings.embed_query(...)` 并打印维度。任何 `import langgraph_state_react` 都会立即产生一次外部 Embedding API 调用与费用。做零成本静态检查时不能导入该模块。

### 4.11 SQLite 运行产物和 checkpoint 区分

根目录有两个用途不同的 SQLite 文件，不能因为它们都以 `.db` 结尾就混为一谈：

```text
short-memory.db
    -> 2026 年 9 月 7 日遗留的本地运行产物
    -> 当前没有脚本写入
    -> 不是当前 HITL 实验使用的数据库

hitl-checkpoint.db
    -> 由 langgraph_human_in_the_loop_sqlite_demo.py 生成
    -> SqliteSaver 的 checkpoint 后端
    -> 保存 Agent State、消息、待审批动作和恢复位置
```

`short-memory.db` 当前大小约 741 KB，最后写入时间为 **2026 年 9 月 7 日 19:38**，
早于后续所有 Context / Middleware 实验。它是孤立的历史遗留产物，是否清理需维护者
明确授权；不要把它当作长期用户资料数据库。

`hitl-checkpoint.db` 已于 **2026 年 9 月 21 日**由 SQLite HITL 实验生成，保存了
`test01` 和 `test02` 两条实验会话的 checkpoint。它同样不是用户长期资料数据库，且
已被 `.gitignore` 的 `*.db` 规则排除，不应提交到 Git。

不要把以下两者混为一谈：

```text
SqliteSaver
  -> 会话 State / 消息 checkpoint
  -> thread_id

SqliteStore
  -> 用户资料和长期业务记忆
  -> user_id 或 namespace + key
```

### 4.12 `langgraph_middleware_wrap_model_call_demo.py`

这是 `wrap_model_call` 的独立学习实验，重点观察模型调用 wrapper 的嵌套关系，以及 wrapper 如何控制下游 `handler` 是否继续执行。

当前文件包含：

- `observe_model_call`：读取 `request.state`、`request.system_message`，并观察下游返回的工具调用和响应元数据。
- `local_cache_shortcut`：当最新用户消息包含“本地缓存”时，直接构造 `ModelResponse`，不调用自己的下游 `handler`。
- 与 `before_model` / `after_model` 的组合：用于观察 wrapper 位于模型节点内部，而 hook 是外部图节点。

维护者已通过真实模型运行完成以下观察：

```text
普通请求：wrapper 继续调用下游，模型可以正常回答并参与 ReAct 工具循环。
缓存请求：命中“本地缓存”后直接返回 [来自本地缓存，未调用模型]，没有工具调用和后续模型轮次。
```

这已经验证了本地短路的代码路径和返回结果。随后加入了独立的最内层 `provider_call_probe`：普通请求进入探针并计数为 1，缓存请求没有进入探针，因此真实模型调用链被短路。`model_call_count` 仍然只能表示进入 `before_model` / 模型节点的次数，不能在短路场景下代表真实 API 请求次数。

当前未完成：

- `handler` 调用 N 次的有限重试。
- 生产级请求缓存、缓存键、过期策略和并发控制。
- 多个同类型 wrapper 组合顺序的系统性实验。

### 4.13 `langgraph_middleware_wrap_tool_call_demo.py`

这是 `wrap_tool_call` 的工具调用观察实验，重点区分“模型生成工具调用意图”和
“工具函数真正执行”两个阶段。工具权限控制已拆分到 4.14 节的独立实验。

当前文件包含：

- `observe_model_call`：辅助观察每轮模型输入消息数、工具调用和响应元数据。
- `observe_tool_call`：接收 `ToolCallRequest`，读取工具名称、参数和 `tool_call_id`，
  在 `handler(request)` 前后打印工具执行过程。
- `build_agent()`：使用 `create_agent` 注册 `get_weather`、`get_user_info` 和
  `save_user_info`，并组合模型观察 wrapper 与工具观察 wrapper；不注册权限 guard。
- `build_store()`：使用 `InMemoryStore` 预置两个用户资料，供用户资料工具精确读取和写入。

拆分前的组合版本已于 **2026 年 9 月 16 日**通过真实模型运行完成基础观察验证：

```text
请求 1（寒暄）：没有 tool_calls，observe_tool_call 未触发。
请求 2（天气和用户资料）：模型分别提出天气和资料工具调用，
每次实际工具调用都进入 observe_tool_call，并返回 ToolMessage。
```

当前版本已移除权限 guard，并将请求 2 简化为“查询资料和天气”。维护者已于
**2026 年 9 月 18 日**重新运行当前独立文件并确认通过，因此上面的工具观察结论
同时适用于拆分后的当前版本。

一次工具观察请求的实际轨迹为：

```text
模型调用 1 -> get_weather / get_user_info tool_call
            -> observe_tool_call -> 真实工具 -> ToolMessage
模型调用 2 -> 读取 ToolMessage -> 最终自然语言回答
```

已确认：

- `wrap_tool_call` 位于工具执行链内部，不是独立的 StateGraph 图节点。
- `handler(request)` 之前可以读取模型生成的工具名称、参数和 `tool_call_id`，
  但这时只有工具调用意图，还不能说明工具函数已经执行。
- `handler(request)` 表示继续下游工具执行链；当前观察 wrapper 调用它后，
  下游最终会执行对应的工具函数并返回 `ToolMessage`。
- 工具结果会回到 Agent 消息 State，并被下一轮模型读取。

当前未完成：

- 工具重试、超时控制和 Human-in-the-loop 审批。
- 多个同类型 `wrap_tool_call` wrapper 组合顺序的系统性实验。

工具异常转换已经拆分到 4.15 节的独立实验；本文件不重复实现该逻辑。

当前限制：

- 使用真实模型运行，工具是否被调用受模型决策影响并产生 API 成本。
- `get_weather` 是固定返回文本的演示工具，不是真实天气服务；资料工具依赖 Store。
- `InMemoryStore` 只在当前 Python 进程内有效。

### 4.14 `langgraph_middleware_tool_guard_demo.py`

这是从工具观察实验中拆出的独立权限控制实验，重点学习工具执行前的权限闸门
以及 `handler(request)` 的 0 次 / 1 次调用差异。

当前文件包含：

- `weather_permission_guard`：读取应用注入的 `runtime.context.authority`，
  不信任模型工具参数中的权限声明。
- `build_agent()`：只注册 `get_weather` 和权限 middleware，去掉 Store、用户资料
  工具以及额外的观察 wrapper，降低实验噪声。
- `main()`：使用完全相同的天气问题，分别以 `admin` 和 `user` 身份执行，
  只改变 `Context.authority`。

拆分前的组合版本已于 **2026 年 9 月 17 日**通过真实模型运行验证权限逻辑：

```text
authority=admin：模型生成 get_weather tool_call -> guard 放行
                -> handler -> 真实 get_weather -> ToolMessage
authority=user：模型生成 get_weather tool_call -> guard 拒绝
                -> 不调用 handler -> 直接返回权限错误 ToolMessage
```

维护者已于 **2026 年 9 月 18 日**重新运行当前独立文件并确认通过。上面的两条
权限路径现在都有独立文件的真实运行凭证，不再只是拆分前组合版本的历史记录。

已确认的机制：

- 权限 wrapper 位于真实工具执行之前。
- 不调用 `handler(request)` 可以跳过下游工具执行链。
- 返回带原始 `tool_call_id` 的 `ToolMessage` 后，Agent 仍可继续下一轮模型处理。
- 权限判断应使用应用注入的 Context，而不是模型自行生成的工具参数。

当前未完成：

- 通用工具权限矩阵、角色继承和审计日志。
- 重试、超时控制和 Human-in-the-loop 审批。

### 4.15 `langgraph_middleware_tool_error_demo.py`

这是 `wrap_tool_call` 的工具异常转换实验，重点学习真实工具抛出 Python 异常后，
如何由 middleware 将异常转换为模型可以继续读取的 `ToolMessage`。

当前文件包含：

- `handle_tool_error`：调用下游 `handler(request)` 执行工具；捕获
  `ZeroDivisionError` 后，构造带原始 `tool_call_id` 的错误 `ToolMessage`。
- `build_agent()`：注册 `divide` 工具和异常处理中间件，并通过系统提示词要求
  所有除法请求都调用工具，包括除数为零的情况。
- `main()`：分别验证正常除法和除零异常两条路径。

维护者已于 **2026 年 9 月 18 日**通过真实模型运行完成以下验证：

```text
请求 1（10 / 2）：模型调用 divide -> 工具返回 5.0 -> ToolMessage -> 最终回答。
请求 2（10 / 0）：模型调用 divide -> divide 抛出 ZeroDivisionError
                -> handle_tool_error 捕获异常 -> 返回错误 ToolMessage
                -> 下一轮模型读取错误并生成最终回答。
```

已确认：

- 工具异常发生在 `handler(request)` 执行下游真实工具的过程中。
- `wrap_tool_call` 可以捕获指定 Python 异常，而不是让 Agent 直接崩溃。
- 异常可以被转换成 `ToolMessage`，并通过原始 `tool_call_id` 与模型的工具调用关联。
- 错误 `ToolMessage` 会进入 Agent 消息 State，触发下一轮模型处理。
- 工具注册并不等于模型必然调用工具；本次异常路径依赖更明确的系统提示词，
  否则模型可能直接回答“除以零未定义”，使异常 middleware 根本不被触发。

当前未完成：

- 对 `ValueError`、`TimeoutError`、`ConnectionError` 等其他明确异常类型的分类处理。
- 工具异常的有限重试、超时控制和 Human-in-the-loop 审批。
- 将异常分类、错误码和用户可见消息抽象成生产级错误协议。

当前限制：

- 当前只对 `ZeroDivisionError` 做了专门转换，不代表所有工具异常都已处理。
- 使用真实模型运行，工具调用决策会产生 API 成本；模型是否生成 tool_call 仍受提示词
  和模型行为影响。
- 当前返回的是教学用固定中文错误信息，尚未建立国际化、脱敏和统一审计策略。

### 4.16 `langgraph_human_in_the_loop_demo.py`

这是 Human-in-the-loop 工具审批实验，重点学习 Agent 在真正执行工具前暂停，
由外部人工决定后再从 checkpoint 恢复执行。

当前文件包含：

- `HumanInTheLoopMiddleware(interrupt_on={"divide": True})`：在 `divide` 工具执行前
  生成中断，不让工具直接运行。
- `InMemorySaver`：保存当前 State、待审批的工具调用和恢复位置；它是短生命周期
  的 checkpoint，不是长期用户资料 Store。
- 第一次 `invoke`：模型提出 `divide` tool_call 后返回 `__interrupt__`，其中包含
  `action_requests` 和 `review_configs`，供审批界面或外部系统展示。
- 第二次 `invoke`：使用相同 `thread_id` 和 `Command(resume={"decisions": [...]})`
  提交人工决定。

维护者已通过真实模型运行验证两条基本路径：

```text
approve：模型生成 divide -> interrupt -> approve -> divide 执行 -> ToolMessage(6.0)
         -> 下一轮模型生成最终回答。

reject：模型生成 divide -> interrupt -> reject -> divide 不执行
        -> ToolMessage 明确说明工具未执行 -> 模型生成未获批准的最终回答，未再次重试。
```

拒绝后恢复时再次看到原来的 AIMessage/tool_call 是正常的：Agent 正在恢复被
checkpoint 暂停的待审批动作。真正判断工具是否执行，应观察后续 `ToolMessage`，而不是
只看 AIMessage。`reject` 拒绝的是当前动作；如果系统提示词允许模型重试，模型仍可能
提出新的 tool_call 并触发新的中断。本实验的系统提示词明确要求拒绝后不要重试。

当前未完成：

- 用 `SqliteSaver` 或 Postgres 等持久化 checkpoint 验证进程重启后的恢复。
- 验证 `edit`、`respond` 等其他人工决定类型。
- 审批身份、超时、审计日志、幂等和审批页面等生产能力。

当前限制：

- `InMemorySaver` 只在当前 Python 进程中有效；换进程或重启脚本后不能恢复这次中断。
- 当前入口主要复现 `reject` 路径，`approve` 已有真实运行记录，但没有在同一次
  `main()` 中同时执行两条分支。
- 使用真实模型运行会产生 API 成本，工具调用是否出现受模型输出和系统提示词影响。

### 4.17 `langgraph_human_in_the_loop_sqlite_demo.py`

这是 4.16 的持久化版本，重点验证 `SqliteSaver` 能否在第一个 Python 进程结束后，
让第二个 Python 进程继续恢复被 Human-in-the-loop 暂停的 Agent。

当前文件包含：

- `DB_PATH`：指向脚本所在目录的 `hitl-checkpoint.db`，避免两个进程因当前工作目录
  不同而打开不同数据库。
- `build_agent(checkpointer)`：接收外部传入的 `SqliteSaver`，保证 `start` 和
  `resume` 使用相同的工具、middleware、系统提示词和图结构。
- `start_agent(thread_id)`：提交原始用户消息，让模型生成 `divide` tool_call，
  在工具真正执行前返回 `__interrupt__`，然后结束当前进程。
- `resume_agent(thread_id, decision)`：重新打开 SQLite 文件，使用相同 `thread_id`
  和 `Command(resume=...)` 恢复，不重新提交原始 `messages`。
- `make_config()` / `make_context()`：分别构造 checkpoint 定位信息和每次运行都要
  重新注入的运行时 `Context`。
- `start` / `resume` 命令行模式：用两个独立 Python 进程模拟“产生中断”和“恢复审批”。

运行方式：

```powershell
python langgraph_human_in_the_loop_sqlite_demo.py start test01
python langgraph_human_in_the_loop_sqlite_demo.py resume test01 approve

python langgraph_human_in_the_loop_sqlite_demo.py start test02
python langgraph_human_in_the_loop_sqlite_demo.py resume test02 reject
```

维护者已于 **2026 年 9 月 21 日**通过真实模型运行验证两条跨进程路径：

```text
test01 / approve：
进程 A 生成 divide -> interrupt -> SQLite 保存 checkpoint
进程 B 使用同一 test01 恢复 -> divide 执行 -> ToolMessage(224.6)
                 -> Agent 完成，没有新的中断。

test02 / reject：
进程 A 生成 divide -> interrupt -> SQLite 保存 checkpoint
进程 B 使用同一 test02 恢复 -> divide 未执行
                 -> 返回“工具未执行”的 ToolMessage
                 -> Agent 生成未获批准的最终回答，没有新的中断。
```

两次恢复都保留了第一次生成的 `tool_call_id`，证明 `resume` 是从 SQLite checkpoint
恢复待审批动作，而不是重新提交用户问题。`HumanMessage` 和原来的 `AIMessage/tool_call`
在恢复输出中再次出现也是正常的恢复轨迹。

必须区分：

```text
SqliteSaver
    -> checkpoint / Agent State / 消息 / 暂停位置
    -> thread_id

SqliteStore
    -> 用户资料和长期业务记忆
    -> user_id 或 namespace + key
```

当前未完成：

- 使用错误的 `thread_id` 做负向恢复验证。
- `edit`、`respond` 决策。
- Postgres 等远程 checkpoint、并发、审批身份、超时、审计、备份和幂等控制。

### 4.18 MCP 天气服务：`langgraph_mcp_demo.py` 与 `mcp_server/get_weather_mcp/server.py`

这是首个把独立 MCP Server 接入 `create_agent` 的端到端学习实验。客户端已从旧的 `MultiServerMCPClient` 迁移到 LangChain 内置的 `MCPAdapter`（通过 `langchain[mcp]` 安装；当前仍为 Beta）：

```text
langgraph_mcp_demo.py
  -> MCPAdapter 的 mcpServers 配置以当前 Python 解释器启动 stdio 子进程
  -> async with 管理 Client/Server 连接生命周期
  -> list_tools() 发现 MCP 工具并适配为 LangChain 工具
  -> create_agent / ainvoke
  -> 模型产生 get_weather tool_call
  -> MCP Client 转发到 FastMCP Server
  -> 天气服务城市查询得到经纬度
  -> 按坐标请求当前天气
  -> 工具结果返回 Agent，模型整理成自然语言
```

当前客户端与服务端实现要点：

- `FastMCP("get_weather_mcp")` 创建服务，`@mcp.tool` 将 `get_weather(city)` 注册为 MCP 能力；服务端只提供工具，不负责决定模型何时调用。
- `langgraph_mcp_demo.py` 从 `langchain.mcp` 导入 `MCPAdapter`，替代旧的 `langchain_mcp_adapters.client.MultiServerMCPClient`；工具发现方法从 `get_tools()` 改为 `await adapter.list_tools()`。
- 适配器配置使用 `{"mcpServers": {"weather": {"command": ..., "args": [...]}}}`；`command` / `args` 描述 stdio 子进程，不再使用旧配置中的 `transport: "stdio"`。
- `async with MCPAdapter(...)` 负责 MCP 连接资源的生命周期。Agent 的 `ainvoke()` 必须在上下文内部执行，因为 Agent 可能在调用期间才使用已发现的 MCP 工具。
- `sys.executable` 用于启动当前 Python 环境中的 MCP Server；`create_agent` 仍负责模型与工具之间的 ReAct 循环，MCPAdapter 只负责 MCP 连接和工具适配。
- `langchain.mcp` 当前为 Beta；`requirements.txt` 通过 `langchain[mcp]>=1.4.2` 安装内置集成，不再声明独立的 `langchain-mcp-adapters`。
- 服务端从仓库根目录 `.env` 读取 `QWEATHER_API_HOST`、`QWEATHER_API_KEY`，以请求头传递密钥；不要在日志或文档中记录凭据值。
- `QWEATHER_API_HOST` 当前按纯 Host 使用，代码会补上 `https://`；配置中若也写 scheme 会形成错误地址。
- 工具先请求城市地理查询，再用返回坐标请求实时天气；当前只查中国范围，同名地点默认取第一个候选。
- HTTP 超时设为 10 秒；HTTP 状态错误和网络错误会转换成可读的工具结果，当前没有自动重试，也未统一处理所有 JSON/schema 异常。
- 返回值只挑选天气状况、温度、体感、湿度和数据归属信息，避免将完整响应 JSON 全部放进模型上下文。
- 维护者于 **2026 年 9 月 26 日**的两次独立运行分别观察到晴间多云（26.59°C、体感 28.79°C、湿度约 59%）和小雨（18°C、体感 17.83°C、湿度约 90%）；输出与湿度按 0–1 比例乘以 100 的代码假设相符，但这些样例不替代完整供应商 Schema 和边界值检查。
- `stdio` 的 stdout 是 MCP 协议通道，服务端不应在该通道打印普通调试日志；需要诊断时应使用 stderr 日志。
- 用户问题中的“今天”目前调用的是实时天气端点，不是逐日天气预报端点。
- 当前运行结果的地点标签显示为“北京市北京”，存在省/市名称重复；属于展示格式小问题，不影响工具查询，可在后续整理输出时修正。

验证状态：维护者于 **2026 年 9 月 26 日**用 `MCPAdapter` 真实运行确认 `get_weather` 工具发现、Agent 生成 `tool_call`、MCP Server 调用天气 API、结果回传和模型基于结果作答。两次独立输出分别记录晴间多云（26.59°C、体感 28.79°C、湿度约 59%）与小雨（18°C、体感 17.83°C、湿度约 90%），均带和风天气 attribution。此前 `It's always sunny ...` 是旧固定天气示例，不作为真实天气验证证据。当前只验证了两次单工具请求，不代表逐日预报、其他城市、异常分支或完整 Schema 已验证。

## 5. 学习进度和路线（Learning Roadmap）

### 5.1 已完成

已完成课程学习和可运行实验的内容包括：

- LLM API 基础和 OpenAI 兼容接口。
- 手写 ReAct、Plan-and-Solve、Reflection 的基本思想和 Demo 阅读。
- LangChain `@tool`、Tool Schema 和结构化 Tool Calling。
- `ToolRuntime`、`Context` 和工具权限控制。
- `create_agent` 高层 Agent API。
- `StateGraph`、State、Node、Edge、条件路由和 `ToolNode`。
- 显式 StateGraph 版 ReAct：`model -> tools -> model`。
- `invoke`、`stream` 和不同流式模式的观察方法。
- StateGraph Mermaid 图的生成，以及静态图和实际运行轨迹的区别。
- 短期记忆、checkpoint、`thread_id` 的概念。
- 长期记忆、`user_id`、Store、结构化用户资料读写。
- `save_user_info` 的增量字典合并：新 key 加入，相同 key 新值覆盖旧值。
- Embedding 和 InMemoryStore 语义检索最小实验。
- `InMemoryStore` 的进程生命周期限制。
- **`SqliteStore` 持久化（课程 Notebook 形式，2026-09-10 维护者确认已完成）**：在 `dive-into-langgraph/6.context.ipynb` 第三节中，用 `sqlite3.connect("user-info.db", check_same_thread=False, isolation_level=None)` 建连接，交给 `SqliteStore(conn)` 作为 Store 后端，用 `store.put(("user_info",), key, value)` 预置资料，再在工具内用 `runtime.store.get(("user_info",), user_id)` 读取。已掌握"**Store 后端可替换、长期资料的生命周期由后端决定**"这一机制。

  **确认程度必须据实表述**（纪律见 3.3 / 6.4.5），不要把它说成"根目录已实测"：本验收只到**课程 Notebook 形式**，根目录实验代码没有落地；本地工作区**不存在** `user-info.db`；且该 Notebook 自带执行输出的 SqliteStore 工具格（第 23 / 26 格）里模型**没有发起工具调用**，直接回答"系统中没有关于她的数据记录"，因此 Notebook 自身也没有留下一次成功的 SqliteStore 读取记录（其输出环境是课程作者的 macOS `/Users/luochang/...`）。当前确认程度是"**机制已学、代码已读**"，不是"本地留有可复现的跨进程读取证据"。

**上下文工程（Context Engineering）阶段已完成的最小实验**：

- **Runtime + Store → Model Context**（`langgraph_context_demo.py`）：从 `runtime.context` 取真实身份、从 `runtime.store` 取长期资料、字段级裁剪后注入本次模型调用的 SystemMessage，且**不写回 State**。
- **State → Model Context**（`langgraph_state_context_demo.py`）：自定义 `AgentState` 扩展字段 `task_mode`，新增 `classify_task` 节点写入 State，模型节点依据 State 切换任务策略提示词。已理解 Context / Store 与 State 在生命周期和语义上的边界差异。
- **`@dynamic_prompt`**（`langgraph_middleware_dynamic_prompt_demo.py`）：在 `create_agent` 中通过 `ModelRequest` 读取 `runtime.context` / `runtime.store` 生成本次系统提示词。已理解它是 `wrap_model_call` 的便捷封装。

**Middleware 阶段已完成的最小实验**：

- **`before_model` / `after_model` 写入 State**（`langgraph_middleware_hooks_demo.py`）：已由**真实模型运行**实测确认两个 hook 被编译成真正的图节点、位于 ReAct 循环内部、返回的 dict 会合并进 State 且 invoke 返回后可读；触发次数等于模型调用次数。两个学习目标均已验证。机制细节见 4.8 与 5.5 节。
- **课程 `3.middleware.ipynb` 第一章"预算控制"已复现**（同一个 `langgraph_middleware_hooks_demo.py`）：`@wrap_model_call` 读 `request.state["messages"]` 长度，超过阈值就 `request.override(model=...)` 切换低费率模型；已实测切换生效（代码阈值 `> 4`）。同时确认了"包裹器在模型节点内部、图节点在外"的层叠关系，以及"能读 State 但不能写 State"这一边界。
- **`wrap_model_call` 短路**（`langgraph_middleware_wrap_model_call_demo.py`）：已通过真实运行命中“本地缓存”分支，直接返回本地 `ModelResponse`，没有工具调用和后续模型轮次；`provider_call_probe` 已验证普通请求计数为 1、缓存请求不进入探针。调用 N 次的重试仍未做。
- **`wrap_tool_call` 工具调用观察**（`langgraph_middleware_wrap_tool_call_demo.py`）：已由真实模型运行验证当前拆分文件中的工具调用前后观察、`handler(request)` 执行真实工具、`ToolMessage` 返回以及多个工具调用的独立触发。异常转换由 4.15 节独立实验负责；重试、超时和审批仍未做。
- **`tool_guard` 权限校验与执行短路**（`langgraph_middleware_tool_guard_demo.py`）：已由真实模型运行验证当前拆分文件中的 `admin` 放行和 `user` 拒绝两条路径。通用权限矩阵、角色继承、审计日志、重试、超时和审批仍未做。
- **`tool_error` 工具异常转换**（`langgraph_middleware_tool_error_demo.py`）：已由真实模型运行验证。正常路径中 `divide(10, 2)` 返回 `5.0`；异常路径中 `divide(10, 0)` 抛出 `ZeroDivisionError`，`handle_tool_error` 捕获后返回错误 `ToolMessage`，Agent 没有崩溃并继续生成最终回答。当前仅覆盖 `ZeroDivisionError`，其他异常、重试和超时仍未做。
- **Human-in-the-loop 基础审批**（`langgraph_human_in_the_loop_demo.py`）：已由真实模型运行验证 `HumanInTheLoopMiddleware` 产生 `__interrupt__`、相同 `thread_id` 恢复，以及 `approve` 执行工具、`reject` 跳过工具并返回拒绝 `ToolMessage` 两条同进程路径。当前文件刻意保留 `InMemorySaver`，用于对照持久化版本。
- **SqliteSaver 跨进程 Human-in-the-loop**（`langgraph_human_in_the_loop_sqlite_demo.py`）：已于 **2026 年 9 月 21 日**由真实模型运行验证。`test01` 在两个独立进程中完成 `approve`，工具返回 `224.6`；`test02` 在两个独立进程中完成 `reject`，工具未执行并返回拒绝 `ToolMessage`，两条路径均没有新的中断。错误 `thread_id`、`edit` / `respond` 和生产级审批能力尚未验证。
- **MCPAdapter 接入真实天气服务**（`langgraph_mcp_demo.py`）：已从 `MultiServerMCPClient/get_tools()` 迁移至 `MCPAdapter/list_tools()`，并通过真实运行验证工具发现、天气调用、工具结果回传及 Agent 最终回答。2026-09-26 两次北京实时天气查询均成功；这只证明单工具调用路径，不代表重复 ReAct、多城市或异常路径已经验证。旧输出中的固定文本 `It's always sunny ...` 不是实时天气数据；地点名重复显示仍待修整。

根据 `dive-into-langgraph` 课程，以下章节标记为已完成：

```text
快速入门
状态图
记忆
上下文
```

后续不应重复堆叠同一种用户资料读写 Demo，而应转向跨组件设计和 Agent 工程能力。

### 5.2 当前阶段与执行顺序

**计划顺序**（5.3 节）与实际执行顺序：

```text
计划:  SqliteStore 持久化验收 -> Context Engineering -> Middleware
实际:  Context Engineering（Runtime+Store / State / @dynamic_prompt）已完成
       -> Middleware（dynamic_prompt / before_model / after_model）已完成，并由真实模型运行实测通过
       -> SqliteStore 持久化验收【已完成，课程 Notebook 形式】
       -> Human-in-the-loop 同进程与 SqliteSaver 跨进程 approve/reject 已验证
       -> MCPAdapter 工具发现、两次真实天气工具调用和结果回传已运行验证（2026-09-26）
```

**SqliteStore 这一项的结论（2026-09-10 维护者确认，原"顺序偏离"记录作废）**：

- 本节曾把"SqliteStore 持久化验收"记为**被跳过、待补做**的真实顺序偏离。维护者确认此项**早已以课程 Notebook 形式完成**（`dive-into-langgraph/6.context.ipynb` 第三节），因此该偏离记录**作废，不再作为欠账**；12.2 节的对应任务已关闭。
- 也就是说学习顺序上**不存在"跳过前置任务"**：Context Engineering 与 Middleware 是正常推进顺序。
- 根目录 `langgraph_context_demo.py`、`langgraph_state_context_demo.py`、`langgraph_middleware_dynamic_prompt_demo.py`、`langgraph_middleware_hooks_demo.py` **仍然全部使用 `InMemoryStore`**。这是**刻意保留，不是未完成**：这四个实验的观察目标是 hook 触发次数、State 写入、以及 Context → Model Context 的构造边界，Store 后端不是当轮的学习变量，换掉它只会引入与学习目标无关的噪声。
- 将来若需要根目录级别的跨进程长期记忆（例如 HITL 恢复实验，或简历项目要真的记住用户资料），**四个实验都通过 `build_store()` 单点注入 Store**，改这一处即可，不需要重写实验。
- 该验收的**确认程度**（本地没有 `user-info.db`、Notebook 自带输出中没有一次成功的 SqliteStore 读取）已在 5.1 节据实记录。**不要把这一项描述成"根目录已实测通过"。**

当前重点：Middleware 的核心实验已经完成，包括模型 wrapper 短路、工具观察、权限
短路和工具异常转换；Human-in-the-loop 的同进程和 `SqliteSaver` 跨进程
`approve` / `reject` 最小路径也已经通过真实模型验证。可选补充 Middleware 的有限
重试或超时控制，但不再阻塞主学习路线。MCP 已迁移到 `MCPAdapter`，并由维护者于 2026 年
9 月 26 日通过两次真实天气请求确认工具发现、调用、结果回传和 Agent 回答。下一阶段进入
RAG；当前仍未验证多城市、逐日预报或天气服务异常分支。

上下文工程阶段需要理解 State、Context、Store、Runtime 的边界，以及如何从它们构造 Model Context、Tool Context 和生命周期上下文。

### 5.3 后续顺序

```text
1. SqliteStore 持久化验收                              [已完成，课程 Notebook 形式；根目录实验刻意保留 InMemoryStore]
2. Context Engineering                                 [已完成]
3. Middleware                                          [核心实验已完成，可选机制待补]
   -> dynamic_prompt                                   [已完成]
   -> before_model / after_model                       [已完成，真实模型实测通过]
   -> wrap_model_call                                  [换模型与短路已验证；重试未做]
   -> wrap_tool_call                                   [已完成：拆分后真实运行通过；重试 / 超时未做]
   -> tool_guard                                       [已完成：拆分后真实运行通过；通用权限矩阵未做]
   -> tool_error                                       [已完成：ZeroDivisionError -> ToolMessage 已验证]
4. Human-in-the-loop
   -> interrupt / 审批 / Command(resume=...)                   [已完成：approve/reject]
   -> SqliteSaver 跨进程暂停、审批和恢复                          [已完成：test01/test02]
5. MCP Server 和外部工具协议
   -> MCPAdapter + stdio Client/Server 工具发现                [已运行验证]
   -> 两次和风天气实时 HTTP 调用与结果回传                    [已运行验证：2026-09-26]
6. RAG：加载、切分、索引、检索、引用
7. Parallelization / Subgraph / Map-Reduce
8. Supervisor / Multi-Agent
9. Evaluation / Observability / Cost / Failure Analysis
10. Production Architecture 和 DeerFlow 源码阅读
```

第 1 步已完成（课程 Notebook 形式），见 5.1 / 5.2 节。当前**不存在未处理的顺序偏离项**，其余步骤保持原计划。

Middleware 放在 Context Engineering 之后，因为中间件经常需要读取和修改 State、Runtime、Prompt、Model 或 Tool 行为。Human-in-the-loop 放在 Middleware 之后，因为中断和恢复依赖可靠的 checkpoint。

### 5.4 DeerFlow 对齐路线

在阅读 DeerFlow 的复杂实现前，先完成显式 StateGraph ReAct、短期 checkpoint、长期 Store、Context、Middleware、HITL、MCP、RAG、Subgraph、并行工作流、Multi-Agent、Sandbox、安全、评估和可观测性等基础。

阅读 DeerFlow 时，优先关注真实源码中的工厂/入口、Lead Agent、Middleware、Task Tool、Subagent Executor、Checkpoint/Client 等职责，并明确区分本仓库已经实现的能力和 DeerFlow 提供的工程化能力。

### 5.5 Middleware 机制备忘（已实际验证）

以下事实来自对已安装 langchain 1.3.14 源码的阅读与本地实测，不是文档推测。学习 Middleware 时应以此为准：

```text
hook                     图形态        执行次数              能否改 State
--------------------------------------------------------------------------
before_agent             节点          整次运行 1 次          能
before_model             节点          每次模型调用前          能（循环入口）
after_model              节点          每次模型调用后          能（循环出口）
after_agent              节点          整次运行 1 次          能
wrap_model_call          包裹器        每次模型调用（可嵌套）   不能，只能改请求/响应
wrap_tool_call           包裹器        每次工具调用           不能，只能改请求/响应
dynamic_prompt           = wrap_model_call 特例，不是独立 hook
```

关键结论：

- **`before_model` / `after_model` 是真正的图节点**，可读写 State，可被 checkpoint，可通过 `can_jump_to` 跳转。
- **`wrap_model_call` / `wrap_tool_call` 不是节点**，是包裹器，只能改写传入的 request / response，**不能修改 State**。
- 多个 `before_model` 按注册顺序串联；多个 `after_model` **逆序**串联（栈式包裹）。
- `state_schema=` 只做声明，不做类型检查；未声明的自定义 State 字段会被**静默丢弃且不报错**。
- `@dynamic_prompt` 由 `wrap_model_call` 实现（源码文档字符串明确说明），因此它 **不写 State**。这是它与 `before_model` 的关键分界。

## 6. 编码规范（Coding Guidelines）

### 6.1 基本原则

- 优先理解原理，再扩展功能。
- 保持学习代码简单、清晰、容易运行和观察。
- 每个实验只引入一个主要新概念。
- 避免过早引入复杂抽象和多层目录。
- 优先使用结构化消息和原生 Tool Calling；新代码不要重新引入 `Thought/Action` 文本正则协议。
- 保留当前有效实验作为对照，不为了代码短而隐藏框架机制。
- 明确区分教学简化实现和生产安全实现。
- 任何外部 API 调用、成本、持久化写入和敏感数据处理都要提前说明。

### 6.2 Agent 核心代码

新 Agent 实验应尽量明确写出：

- 输入是什么。
- State 保存什么。
- Context 保存什么。
- Store 保存什么。
- 模型节点产生什么。
- 工具节点如何执行。
- 条件路由如何决定继续或结束。
- 失败时状态如何变化。
- 最终输出从哪里读取。

### 6.3 工具规范

- 工具使用 `@tool` 或明确的 `BaseTool`。
- 工具必须有准确的 docstring 和类型注解。
- 参数应是模型容易生成和校验的结构化类型。
- 权限检查必须位于工具或可靠的运行时策略中，不能只写在 Prompt 里。
- 外部 API 错误应与正常业务结果区分，不能把错误文本伪装成成功结果。
- 工具应尽量短小、单一职责，并明确副作用。
- 工具不应让模型任意指定真实用户身份或越权访问其他用户数据。

### 6.4 注释和文档规范（重要）

#### 6.4.1 核心原则

仓库维护者明确反馈过：**"代码注释不够详细，我看不懂"**。因此本仓库的注释标准高于一般生产项目——这里的代码首先是**教材**，其次才是可运行程序。

判定标准只有一条：

> **一个只有后端 CRUD 经验、正在学 Agent 的人，能否只靠注释读懂这段代码在做什么、为什么这么做、有什么限制。**

注释必须解释 **Why / What / How / Limitations**，而不只是复述代码行为：

```text
Why          为什么需要这一步？不这样做会出什么问题？
What         这段代码在业务/框架语义里代表什么？
How          与后端概念如何对应？（State≈流程图状态，Node≈Service，
             Checkpoint≈可恢复快照，Tool≈受控外部接口）
Limitations  教学简化在哪？生产中需要补什么？
```

#### 6.4.2 反面与正面示例

反面（禁止，等于没写）：

```python
# 读取 store
user_item = runtime.store.get(("users",), user_id)
# 判断是否为空
profile = user_item.value if user_item else {}
```

正面（要求达到的水平）：

```python
# 从长期记忆中按 user_id 精确读取该用户资料。
# 用精确 key 查询而不是语义 search()，因为这是结构化事实（姓名/语言），
# 需要 100% 准确，不能接受向量检索的近似排序。
user_item = runtime.store.get(("users",), user_id)

# Store 中没有该用户时返回 None，这里退化成空字典，
# 让后续 profile.get("language", 默认值) 能安全取默认值，
# 而不是让整个模型节点因 KeyError 崩溃。
profile = user_item.value if user_item else {}
```

#### 6.4.3 文件头 docstring（每个实验文件必须有）

参照 `embedding_test.py` 的现有写法，这十项**缺一不可**：

```text
实验名称：
实验目标：
解决的问题：            <- 为什么需要这个概念，解决什么痛点
使用的 Agent 概念：
系统架构：              <- 用箭头文字图画出数据流
实现方式：
验证方式：              <- 具体看什么输出、什么叫通过
学习总结：
已知限制：              <- 当前实现的不足
后续优化方向：
```

若某项尚未验证，必须写"未验证"，**不得省略或编造**。

#### 6.4.4 必须写详细注释的位置

以下位置**不允许只有一行注释**，必须成段说明：

1. **框架高层 API 背后的机制**。使用 `create_agent`、`@dynamic_prompt`、`@before_model` 等高层封装时，必须说明它隐藏了哪些 LangGraph 底层动作。例如必须写明 `@dynamic_prompt` 实际由 `wrap_model_call` 实现、不产生图节点、不写 State。
2. **State / Context / Store / Runtime 的职责边界**。每次出现这四者中的任意一个，都要说明它是什么生命周期、由谁写入、能活多久。
3. **反直觉的设计决策**。例如"动态 SystemMessage 故意不写回 `state["messages"]`"，必须解释如果写回会导致循环内重复累积。
4. **易踩的静默陷阱**。例如 hook 返回未声明的 State 字段会被**静默丢弃且不报错**——必须显式警告。
5. **外部 API 调用点**。哪些行会真正发出网络请求、产生费用、需要 `.env`。
6. **导入即产生副作用的模块**。例如某些文件在模块加载阶段就调用 Embedding 或 `invoke`，必须标注"import 本模块会立即产生外部 API 调用"。
7. **教学简化 vs 生产实现**。硬编码关键词分类、未处理并发写、无超时重试等，都要标注清楚。

#### 6.4.5 注释粒度

- **不要给每一行加注释**，也不要写 `# 判断是否为 None` 这类复述。
- **不要用大量注释掩盖过于复杂的实现**——复杂度过高时应先拆分代码。
- 注释与代码必须同步修改。**代码改了、注释没改，比没有注释更糟**（`embedding_test.py` 中 1024 / 2048 / 4096 三个维度数字互相矛盾，就是这个问题的真实案例）。
- 中文注释为主，框架专有名词保留英文原名（如 `StateGraph`、`checkpoint`、`reducer`），便于对照官方文档检索。

#### 6.4.6 命名规范

- State 类名要表达**数据内容**而不是"给某功能用的"。例如 `MiddlewareState`（说明用途）优于 `HookState`（含义模糊，曾被维护者质疑）。
- 不解释清楚来源的类型名不要用。使用 `AgentState`、`MessagesState` 这类框架类型时，注释要说明它**来自哪个包、有哪些自带字段**，因为维护者会问"这个类我没定义，怎么就能继承"。

#### 6.4.7 其他文档要求

重要实验应同步维护 README、Notebook Markdown 或文件头独立说明，至少记录实验目标、使用的 Agent 概念、系统架构、关键设计决策、验证方式、当前限制和后续方向。

修改代码后，必须同步更新所有变旧的注释与本文档相关章节。

## 7. 实验规范（Experiment Guidelines）

每个新实验应记录：

```text
实验名称：
实验目标：
解决的问题：
使用的 Agent 概念：
系统架构：
实现方式：
验证方式：
学习总结：
后续优化方向：
```

推荐实验粒度：

```text
LLM API
  -> Tool Calling
  -> 单工具 Agent
  -> StateGraph ReAct
  -> Checkpoint / Memory
  -> Context Engineering
  -> Middleware
  -> Human-in-the-loop
  -> RAG / MCP
  -> Workflow / Subgraph
  -> Multi-Agent System
```

每个实验都应提供至少一个可观察验收点，例如：

- 打印模型实际生成的 tool call。
- 打印工具收到的参数和返回的 ToolMessage。
- 打印节点流转顺序。
- 对比不同 `thread_id` 和 `user_id` 的结果。
- 验证重启前后的 Store 行为。
- 使用固定输入验证路由和权限边界。

## 8. AI Coding Agent 工作方式

未来协助本仓库的 AI Agent 应：

- 先阅读实际文件、依赖和运行结果，再提出修改。
- 扮演技术导师，而不是只生成代码。
- 先说明目标、假设、影响文件、风险和验证方式。
- 解释架构设计、技术取舍和与后端经验的对应关系。
- 维护者要求“看代码/过目/解释”时只读检查，不擅自修改。
- 维护者明确要求修改时，只修改任务范围内的文件。
- 保留维护者已有改动，不使用破坏性 Git 命令覆盖文件。
- 删除文件前必须得到明确授权，并核对精确目标；不得因为“清理”而删除课程资料、Skill、环境或数据库。
- 不读取、打印或提交 `.env` 中的密钥。
- 对照成熟框架时，区分本仓库当前实现、课程示例和框架官方能力。
- 修改后运行与当前仓库匹配的最小验证；如果没有测试框架，明确说明只完成了语法检查或手动验证。
- 不凭空声称已经执行模型调用、测试或支持某项能力。

## 9. 简历项目演进方向（Portfolio Evolution）

当前仓库仍是多个学习实验的集合，最终业务场景已经初步确定为 RepoResearcher，但尚未进入实现阶段。最终项目不应只是许多互相独立的 Demo，而应逐步形成一个完整、可解释、可评估的 Agent 应用。

RepoResearcher 的定位是面向代码仓库的研究与研发辅助 Agent：未来帮助开发者分析陌生仓库、定位功能入口、追踪调用链、执行受控验证并生成带源码证据的技术报告。当前只记录方向，不把它描述成已实现能力；在正式开发前继续完成 Middleware、Human-in-the-loop、MCP、RAG、Subgraph、并行、Multi-Agent、Evaluation 和 Observability 等基础。

建议演进路径：

1. 选择一个真实业务问题和明确用户。
2. 把当前 Agent Loop、工具注册和运行时配置整理成可复用模块。
3. 接入真实外部数据源或业务 API。
4. 加入短期状态持久化、长期记忆、会话隔离和权限控制。
5. 根据场景加入 Context Engineering、RAG、Workflow 或 Human-in-the-loop。
6. 使用 MCP 或标准化接口扩展工具能力。
7. 设计评估数据集、基线和质量指标。
8. 加入 Trace、错误分析、Token/成本统计和回归测试。
9. 补充部署、配置、安全和运维文档。
10. 对照 DeerFlow 说明哪些能力自己实现、哪些能力由框架提供。

项目最终文档应回答：

- 解决了什么真实问题？
- 为什么需要 Agent，而不是普通 CRUD 或固定流程？
- Agent 的 State、Context、Tool、Workflow 和权限边界是什么？
- 如何证明系统质量？
- 失败、超时、幻觉和工具错误如何处理？
- 如何部署、监控和维护？

## 10. 项目开发规则（Repository Rules）

### 10.1 代码组织

- 当前仓库保持平铺式学习脚本结构，不要在没有必要时引入复杂包结构。
- 新增 Python 模块优先使用 `snake_case.py`。
- 工具定义集中在 `tools.py` 或明确的工具模块，Agent 图和运行入口集中在 Agent 模块。
- 新实验可以先独立成文件；重复出现且稳定后再抽取公共模块。
- 课程资料目录 `dive-into-langgraph/` 和本地 Skill 目录 `.agents/skills/dive-into-langgraph/` 只读参考，不作为根项目业务代码随意整理。
- 不要把 API 密钥、真实账号、持久化数据或隐私信息写入代码和 Notebook 输出。

### 10.2 测试和验证

- 当前没有测试框架或测试目录，不能假设存在 `pytest` 命令。
- 新增复杂逻辑时，优先为状态路由、工具参数校验、权限边界和失败路径设计可重复测试。
- Agent API 调用测试应尽量使用 Fake Model、Mock Tool 或固定数据，避免每次测试产生外部调用和费用。
- 手动运行模型的结果不能替代确定性测试，必须标记为手动验证。

### 10.3 安全

- `.env` 只用于本地配置，禁止打印和提交。
- 不要在宿主进程直接执行不可信的模型生成代码。
- 新增计算工具时避免 `eval()`；学习目的若必须使用，也要限制输入并明确不适用于生产。
- 外部工具应有超时、错误分类、权限控制和必要的输出大小限制。
- 涉及用户资料的工具应遵循最小权限原则，并使用运行时身份而非模型传入的身份。

### 10.4 持久化边界

```text
State / MessagesState
    -> 当前工作流状态

Checkpointer / SqliteSaver
    -> 按 thread_id 保存短期会话和图状态

Context / ToolRuntime
    -> 本次运行的身份、权限和环境

Store / SqliteStore
    -> 按 user_id 或 namespace + key 保存长期业务资料
```

不要因为它们都可能使用 SQLite，就把 checkpoint、运行时上下文和长期用户资料混存或混称。

## 11. 开发命令（Development Commands）

仓库当前没有官方安装、启动、测试、构建或部署脚本。以下命令只记录当前实际使用的本地入口，不应扩展成不存在的命令。

### 11.1 语法检查

```powershell
.\.venv\Scripts\python.exe -m py_compile tools.py langgraph_react.py langgraph_state_react.py embedding_test.py langgraph_context_demo.py langgraph_state_context_demo.py langgraph_middleware_dynamic_prompt_demo.py langgraph_middleware_hooks_demo.py langgraph_middleware_wrap_model_call_demo.py langgraph_middleware_wrap_tool_call_demo.py langgraph_middleware_tool_guard_demo.py langgraph_middleware_tool_error_demo.py langgraph_mcp_demo.py mcp_server/get_weather_mcp/server.py
```

### 11.2 LangGraph Agent 实验

```powershell
.\.venv\Scripts\python.exe langgraph_react.py
```

```powershell
.\.venv\Scripts\python.exe langgraph_state_react.py
```

这些脚本会访问外部 LLM/Embedding API，运行前必须确认 `.env` 和网络可用，并注意 API 成本。`langgraph_react.py` 在修正必填的 `Context.user_id` 之前，不应描述成已完成运行验证。

### 11.3 Embedding 和长期资料实验

```powershell
.\.venv\Scripts\python.exe embedding_test.py
```

该脚本会访问 Embedding 和 Chat API；它是手动实验，不是正式自动化测试。

### 11.4 MCP 天气 Agent

```powershell
.\.venv\Scripts\python.exe langgraph_mcp_demo.py
```

该入口会启动 MCP stdio 子进程、调用真实天气服务并调用 LLM。运行前需配置 `.env` 中的
`LLM_MODEL_ID`、`LLM_API_KEY`、`LLM_BASE_URL`、`QWEATHER_API_HOST` 和 `QWEATHER_API_KEY`；
会产生外部请求，静态检查时不要运行它。

### 11.5 课程目录

课程 Notebook、脚本和项目配置位于 `dive-into-langgraph/` 内，应使用已有 IDE 或已配置的 Notebook 环境打开。不要把课程目录的命令未经验证地写成根项目命令。

### 11.6 测试和构建

- 当前没有正式测试命令。
- 当前没有构建命令。
- 当前没有部署命令。
- 新增这些能力后，必须同步更新本文件和项目说明。

## 12. 当前待办（TODO）

### 12.1 当前学习任务（最高优先级）

- **`before_model` / `after_model` 实验已完成**（`langgraph_middleware_hooks_demo.py`）：hook 已真正写入 State，两个学习目标均已由**真实模型运行**实测通过（触发次数 = 模型调用次数；返回的 dict 合并进 State 且 invoke 返回后可读）。文件头已按 6.4.3 补齐十项 docstring，基类为 `AgentState` 的扩展 `CallCountState`。详见 4.8 节。
- 该实验**刻意只用真实模型验证，不引入假模型 / Mock**（维护者明确要求）。因此每次验证都会产生真实 API 调用与费用，结果受模型当时行为影响，不是确定性的。文件内没有自检路径。
- **`wrap_model_call` 已完成的部分**：课程的"换模型"用法（`request.override(model=...)`）、`request.state` 只读读取、`handler(request)` 才真正发起调用——均已随课程"预算控制"复现验证，见 4.8 / 5.1 节。
- **`wrap_model_call` 的短路已完成初步验证**（`langgraph_middleware_wrap_model_call_demo.py`）：真实运行中通过消息命中“本地缓存”条件，直接得到 `[来自本地缓存，未调用模型]`，没有工具调用和后续模型轮次。随后加入 `provider_call_probe`，普通请求进入探针并计数为 1，缓存请求未进入探针；调用 N 次的重试仍未做。
  - **判据警告**：`before_model` 的计数不能证明“模型没被调用”，因为它在 wrapper 上游；应结合 wrapper/provider 探针和响应元数据判断。
- **`wrap_tool_call` 工具观察与 `tool_guard` 权限控制已完成**：维护者于 2026 年 9 月 18 日重新运行两个拆分后的独立文件，确认工具观察、`handler(request)` 执行工具、`admin` 放行和 `user` 短路路径均通过。重试、超时、通用权限矩阵和审批仍未做。
- **`tool_error` 工具异常转换已完成最小实验**（`langgraph_middleware_tool_error_demo.py`）：真实运行中验证 `divide(10, 0)` 产生 `ZeroDivisionError`，middleware 将其转换为错误 `ToolMessage`，Agent 继续完成下一轮模型处理。下一步可扩展明确异常类型的分类处理，再考虑有限重试；重试必须限定异常类型、次数和退避策略，不能对所有异常无限重试。
- **Human-in-the-loop 基础审批已完成**（`langgraph_human_in_the_loop_demo.py`）：2026 年 9 月 18 日真实运行验证了 `__interrupt__`、同一 `thread_id` 恢复、`approve` 执行工具和 `reject` 返回“工具未执行”的 `ToolMessage`，且当前提示词下拒绝后没有新的工具调用。`edit` / `respond`、持久化 checkpoint、审批身份、超时和审计仍未做。
- **SqliteSaver 跨进程 Human-in-the-loop 已完成**（`langgraph_human_in_the_loop_sqlite_demo.py`）：2026 年 9 月 21 日真实运行验证 `test01` 的 approve 和 `test02` 的 reject。两个独立 Python 进程使用同一个 `hitl-checkpoint.db` 和同一个 `thread_id` 恢复成功；approve 执行 `divide` 并返回 `224.6`，reject 返回工具未执行的 `ToolMessage`，两条路径均没有新的中断。错误 `thread_id`、`edit` / `respond`、审批身份、超时和审计仍未做。
- **MCPAdapter 真实天气调用已运行验证**（`langgraph_mcp_demo.py` + `mcp_server/get_weather_mcp/server.py`）：2026 年 9 月 26 日，维护者使用 `MCPAdapter` 成功发现并调用 `get_weather` 两次，工具结果均返回给 Agent 并形成最终回答。观测样例包括晴间多云（26.59°C、体感 28.79°C、约 59% 湿度）和小雨（18°C、体感 17.83°C、约 90% 湿度）。仅覆盖两次单工具请求，不扩展为多工具循环、其他城市、逐日预报或错误路径已验证；地点名仍有重复显示。

### 12.2 Store 持久化验收（已关闭）

- **SqliteStore 持久化验收已完成**（2026-09-10 维护者确认）：以课程 Notebook 形式完成，即 `dive-into-langgraph/6.context.ipynb` 第三节（`sqlite3.connect("user-info.db")` + `SqliteStore(conn)` + `runtime.store.get(...)`）。原记录的"被跳过、需补做"作废。见 5.1 / 5.2 节。
- 根目录 `langgraph_context_demo.py`、`langgraph_state_context_demo.py`、`langgraph_middleware_dynamic_prompt_demo.py`、`langgraph_middleware_hooks_demo.py` **刻意保留 `InMemoryStore`**，不是未完成项；四个文件的 `build_store()` 都是 Store 的单点注入位置，将来需要跨进程持久化时只改这一处。
- **确认程度（据实记录）**：本地工作区不存在 `user-info.db`，该 Notebook 自带的执行输出中也没有一次成功的 SqliteStore 读取。因此这一项的结论是"机制已学、代码已读"，不是"根目录已实测"。详见 5.1 节。
- **仍然未验证的组合**：`SqliteStore` 与 `before_model` / `after_model` hook 一起工作（即 hook 能否读到跨进程持久化的长期资料）。这个组合从未做过；若将来需要，可作为独立小实验。

### 12.3 已知缺陷与清理

- 修正 `langgraph_react.py` 第 38 / 47 行缺少 `user_id` 的 `Context` 调用。该文件目前在 import 阶段即抛 `ValidationError`，并会立即发起真实模型调用。
- 清理 `langgraph_react.py` 的未使用导入。
- 清理 `langgraph_state_react.py` 中结构化资料实验不需要的 Embedding 探针（当前在 import 阶段就产生外部 API 调用与费用）和未使用导入。
- 统一 `embedding_test.py` 中与实际 4096 维输出不一致的旧注释（第 12 / 19 / 81 行分别为 1024 / 1024 / 2048，与第 90 行代码矛盾）。
- 清理 `langgraph_context_demo.py`、`langgraph_state_context_demo.py` 中未使用的导入（`sqlite3`、`SqliteSaver`、`InMemorySaver`、`OpenAIEmbeddings`）。

### 12.4 工程与文档基础设施

- README.md 已创建，记录实验顺序、架构图、运行前提、真实进度和已确定但尚未实现的 RepoResearcher 方向。
- 为 StateGraph 路由、工具权限、资料合并、Store 隔离和天气响应解析建立最小测试，避免每次验证都产生外部调用和费用（原则见 10.2 节）。当前仓库**尚未建立**任何此类测试；已有的 Context / Middleware 实验全部采用真实模型手动验证，不使用假模型 / Mock。

### 12.5 后续学习任务

- 使用错误 `thread_id` 完成 Human-in-the-loop 的跨会话负向恢复验证。
- 验证 Human-in-the-loop 的 `edit` / `respond` 决策，并补充审批身份、超时、审计和幂等设计。
- 学习 RAG：从课程的文档加载、切分、索引和检索示例开始，再比较向量检索、关键词检索、混合检索与引用；先做一个能检查检索结果的小实验。
- 后续再学习 Subgraph、并行和 Supervisor/Multi-Agent。
- 选择最终简历项目的真实业务场景。
- 建立评估数据、Trace、成本统计和失败分析机制。
- 在具备足够基础后，开始按组件阅读 DeerFlow 源码，并记录架构对照笔记。

### 12.6 待决策事项

- `short-memory.db`（741 KB，孤立文件，最后写入 2026-09-07）是否删除。**删除前必须得到维护者明确授权**，见 4.11 节。
