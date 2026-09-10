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
├── langgraph_middleware_hooks_demo.py                  # before_model / after_model（实验进行中）
├── requirements.txt
├── short-memory.db       # 本地 SQLite 运行产物；当前无脚本写入，见 4.11 节
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

当前根目录没有正式测试目录、CI 配置、构建配置、部署配置或 README.md。不要凭空假设这些设施存在。

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

### 3.4 版本控制现状（风险提示）

截至 2026 年 9 月 10 日：

```text
根目录 hello-agents/      -> 不是 git 仓库，没有 .git，也没有 .gitignore
dive-into-langgraph/      -> 是独立的 git 仓库（课程源码），与根项目实验代码无关
```

影响：

- 根目录实验代码**没有任何版本历史**，修改后无法通过 git 回滚。
- 修改重要文件前，应先自行备份，或先完成“新增 `.gitignore` + 建立 git 仓库”这一 TODO。
- **不要**在根目录用 git 命令操作 `dive-into-langgraph/`，它们是两个独立仓库。
- `dive-into-langgraph/` 内已存在的未提交修改属于课程内容，除非维护者明确要求，不要提交或还原。

## 4. 技术栈、配置和实验代码

### 4.1 技术栈和依赖

项目主要使用：

- Python。
- `langchain`：Agent API、消息和工具抽象。
- `langchain-openai`：通过 OpenAI 兼容接口创建 `ChatOpenAI` 和 `OpenAIEmbeddings`。
- `langgraph`：StateGraph、ToolNode、checkpoint 和 Store 运行时能力。
- `pydantic`：`Context` 等运行时数据结构。
- `python-dotenv`：加载 `.env`。
- `google-search-results`：为 `tools.py` 提供 SerpApi 客户端。
- `openai`：OpenAI 兼容 SDK 的底层依赖。
- `ipython`：消息展示和 Notebook 支持。

`requirements.txt` 当前明确声明：

```text
openai
python-dotenv
google-search-results
ipython
langchain>=1.0
langchain-openai
```

当前本地环境曾实际观察到 Python 3.13.1、LangChain 1.3.14、LangChain OpenAI 1.4.1、LangGraph 1.2.10。这些是本地环境事实，不是项目锁定版本。

当前代码直接使用了 `langgraph` 和 `pydantic`，但它们尚未在根目录 requirements.txt 中作为直接依赖明确声明。未来整理依赖时应补充，而不是依赖传递安装。

### 4.2 环境变量

`.env` 中使用过以下配置名称，文档、日志和回答中不得输出其值：

- `LLM_MODEL_ID`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `EMBEDDING_MODEL_ID`
- `MODEL_PROVIDER`
- `SERPAPI_API_KEY`

仓库当前没有根目录 `.gitignore`。建立版本控制或提交代码前，应至少排除：

```text
.env
.venv/
__pycache__/
.idea/
.omo/
*.db
```

### 4.3 `tools.py`

`tools.py` 是当前共享工具模块，包含：

- `search(query)`：通过 SerpApi 搜索互联网，并返回答案框、知识图谱或有机搜索摘要。
- `calculate(runtime, a, b)`：通过 `ToolRuntime[Context, Any]` 读取权限，仅允许 `admin` 使用加法工具。
- `get_weather(city)`：固定返回演示天气文本，不是真实天气服务。
- `get_user_info(runtime)`：从 `runtime.store` 按 `("users",)` 和当前 `user_id` 精确读取用户资料。
- `save_user_info(user_info, runtime)`：读取旧资料，将新资料合并后写回 Store；相同 key 由新值覆盖旧值。

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

### 4.8 `langgraph_middleware_hooks_demo.py`：`before_model` / `after_model`（进行中）

当前状态：**实验进行中，尚未完成验证。**

- 文件已存在并包含 `build_store()` / `build_agent()` / `main()` 骨架，以及注册好的 `@before_model` 与 `@after_model`。
- 但两个 hook 目前都 `return None`，只做 `print`，**没有真正写入 State**，因此尚未触及 Middleware 最核心的能力。
- 已执行过的验证：模块可导入、`build_agent(store)` 可构建、`draw_mermaid()` 可输出图结构（节点为 `{middleware名}.before_model` 与 `{middleware名}.after_model`）。
- **尚未执行任何真实模型调用**，hook 的运行时行为未验证。
- 已知待修正：草稿从 `langgraph.graph` 导入 `MessagesState`，但 `create_agent` 的 hook 应使用 `langchain.agents.middleware.AgentState` 作为基类（`create_agent` 依赖 `AgentState` 的 `jump_to` / `structured_response` 字段）。

`before_model` / `after_model` 在 `create_agent` 内部的真实位置（依据 `langchain/agents/factory.py` 源码）：

```text
START
  -> [before_agent ...]        整次运行只跑一次
  -> before_model              循环入口：每次调用模型前都跑
  -> model
  -> after_model               每轮迭代出口
  -> 条件路由 ─┬─ tools -> 回到 before_model
               └─ [after_agent] -> END
```

关键源码事实：

- hook 被编译为真正的图节点：`graph.add_node(f"{m.name}.before_model", ...)`。
- 循环入口为第一个 `before_model`：`loop_entry_node = f"{middleware_w_before_model[0].name}.before_model"`。
- 多个 `after_model` **逆序**串联（`range(len-1, 0, -1)`），因为它是栈式包裹。
- `state_schema=` 是装饰器的关键字参数（`before_model(func=None, *, state_schema=None, tools=None, can_jump_to=None, name=None)`）。它只做声明，不做类型检查。

**必须记住的坑**：hook 返回的自定义 State 字段**必须**出现在合并后的 State schema 中，否则会被**静默丢弃且不报错**。已实测：未声明 `state_schema` 时返回 `{"model_call_count": 99}`，最终读出 `None`；声明后读出 `99`。

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

### 4.11 `short-memory.db`

这是本地 SQLite 运行产物，当前大小约 741 KB，最后写入时间为 **2026 年 9 月 7 日 19:38**，早于后续所有 Context / Middleware 实验。

**当前没有任何脚本在写入它**：全仓库的 `SqliteSaver` 只有 `langgraph_state_react.py` 第 86–87 行以注释形式存在，其余文件仅保留未使用的 import。因此：

- 不要因为文件存在就推断 checkpoint 持久化"正在使用中"。
- 不要把它当作长期用户资料数据库。
- 它是孤立的历史遗留产物，是否清理需维护者明确授权。

不要把以下两者混为一谈：

```text
SqliteSaver
  -> 会话 State / 消息 checkpoint
  -> thread_id

SqliteStore
  -> 用户资料和长期业务记忆
  -> user_id 或 namespace + key
```

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

**上下文工程（Context Engineering）阶段已完成的最小实验**：

- **Runtime + Store → Model Context**（`langgraph_context_demo.py`）：从 `runtime.context` 取真实身份、从 `runtime.store` 取长期资料、字段级裁剪后注入本次模型调用的 SystemMessage，且**不写回 State**。
- **State → Model Context**（`langgraph_state_context_demo.py`）：自定义 `AgentState` 扩展字段 `task_mode`，新增 `classify_task` 节点写入 State，模型节点依据 State 切换任务策略提示词。已理解 Context / Store 与 State 在生命周期和语义上的边界差异。
- **`@dynamic_prompt`**（`langgraph_middleware_dynamic_prompt_demo.py`）：在 `create_agent` 中通过 `ModelRequest` 读取 `runtime.context` / `runtime.store` 生成本次系统提示词。已理解它是 `wrap_model_call` 的便捷封装。

**Middleware 阶段进行中**：

- `before_model` / `after_model` 已在 `langgraph_middleware_hooks_demo.py` 中注册并确认可编译成图节点，但**尚未写入 State，也未经过真实模型调用验证**。详见 4.8 节。

根据 `dive-into-langgraph` 课程，以下章节标记为已完成：

```text
快速入门
状态图
记忆
```

后续不应重复堆叠同一种用户资料读写 Demo，而应转向跨组件设计和 Agent 工程能力。

### 5.2 当前阶段与执行顺序偏离（重要）

**计划顺序**（5.3 节）与实际执行顺序存在偏离，如实记录如下：

```text
计划:  SqliteStore 持久化验收 -> Context Engineering -> Middleware
实际:  Context Engineering（Runtime+Store / State / @dynamic_prompt）已完成
       -> Middleware（before_model / after_model）进行中
       -> SqliteStore 持久化验收【被跳过，尚未执行】
```

**偏离事实（截至 2026 年 9 月 10 日）**：

- SqliteStore 持久化验收**没有执行**。`langgraph_context_demo.py`、`langgraph_state_context_demo.py`、`langgraph_middleware_dynamic_prompt_demo.py`、`langgraph_middleware_hooks_demo.py` **全部仍在使用 `InMemoryStore`**，因此所有长期资料实验的数据都只在单个 Python 进程内有效，退出即丢失。
- 这是**真实的顺序偏离，不是文档表述问题**。前一轮 Agent 直接进入了 Context Engineering，未先完成跨进程持久化验收。

**待决策**：SqliteStore 持久化验收应作为独立实验补做，还是在 Middleware 阶段告一段落后补做。补做本身不是新概念，只是把 Store 后端从 `InMemoryStore` 换成 `SqliteStore`，并验证"第一次运行保存、结束 Python、第二次运行用新 `thread_id` 与相同 `user_id` 仍能读取"。

当前重点：完成 `before_model` / `after_model` 实验（让 hook 真正写入 State 并验证触发时机）。

上下文工程阶段需要理解 State、Context、Store、Runtime 的边界，以及如何从它们构造 Model Context、Tool Context 和生命周期上下文。

### 5.3 后续顺序

```text
1. SqliteStore 持久化验收                              [未执行，被跳过，待补做]
2. Context Engineering                                 [已完成]
3. Middleware                                          [进行中]
   -> dynamic_prompt                                   [已完成]
   -> before_model / after_model                       [进行中，当前重点]
   -> wrap_model_call                                  [未开始]
   -> wrap_tool_call                                   [未开始]
4. Human-in-the-loop
   -> interrupt
   -> 审批
   -> Command(resume=...)
5. MCP Server 和外部工具协议
6. RAG：加载、切分、索引、检索、引用
7. Parallelization / Subgraph / Map-Reduce
8. Supervisor / Multi-Agent
9. Evaluation / Observability / Cost / Failure Analysis
10. Production Architecture 和 DeerFlow 源码阅读
```

第 1 步被跳过的事实见 5.2 节。它是唯一的顺序偏离项，其余步骤保持原计划。

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

当前仓库仍是多个学习实验的集合，尚未确定最终业务场景。最终项目不应只是许多互相独立的 Demo，而应逐步形成一个完整、可解释、可评估的 Agent 应用。

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
.\.venv\Scripts\python.exe -m py_compile tools.py langgraph_react.py langgraph_state_react.py embedding_test.py
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

### 11.4 课程目录

课程 Notebook、脚本和项目配置位于 `dive-into-langgraph/` 内，应使用已有 IDE 或已配置的 Notebook 环境打开。不要把课程目录的命令未经验证地写成根项目命令。

### 11.5 测试和构建

- 当前没有正式测试命令。
- 当前没有构建命令。
- 当前没有部署命令。
- 新增这些能力后，必须同步更新本文件和项目说明。

## 12. 当前待办（TODO）

### 12.1 当前学习任务（最高优先级）

- **完成 `before_model` / `after_model` 实验**（`langgraph_middleware_hooks_demo.py`，当前重点）：让两个 hook 真正返回 dict 写入 State，并验证触发时机（触发工具的请求各触发 2 次，不触发的各 1 次）。详见 4.8 节与 5.5 节。
- 该文件同时需要修正：把 `from langgraph.graph import MessagesState` 改为使用 `langchain.agents.middleware` 的 `AgentState` 作为自定义 State 基类。
- 为该文件补齐符合 6.4.3 要求的文件头 docstring（十项齐全）。

### 12.2 被跳过的前置任务（需补做）

- **SqliteStore 持久化验收**：将结构化资料的 Store 后端从 `InMemoryStore` 迁移到 `SqliteStore`，完成跨 Python 重启测试（第一次运行保存 `user_3`，结束进程，第二次运行用新 `thread_id` 与相同 `user_id` 仍能读取）。当前 4 个 Context / Middleware 实验全部仍是 `InMemoryStore`，数据退出即丢失。见 5.2 节。

### 12.3 已知缺陷与清理

- 修正 `langgraph_react.py` 第 38 / 47 行缺少 `user_id` 的 `Context` 调用。该文件目前在 import 阶段即抛 `ValidationError`，并会立即发起真实模型调用。
- 清理 `langgraph_react.py` 的未使用导入。
- 清理 `langgraph_state_react.py` 中结构化资料实验不需要的 Embedding 探针（当前在 import 阶段就产生外部 API 调用与费用）和未使用导入。
- 统一 `embedding_test.py` 中与实际 4096 维输出不一致的旧注释（第 12 / 19 / 81 行分别为 1024 / 1024 / 2048，与第 90 行代码矛盾）。
- 清理 `langgraph_context_demo.py`、`langgraph_state_context_demo.py` 中未使用的导入（`sqlite3`、`SqliteSaver`、`InMemorySaver`、`OpenAIEmbeddings`）。

### 12.4 工程与文档基础设施

- 将 `langgraph`、`pydantic` 等直接使用的依赖补充到 `requirements.txt`。
- 新增根目录 `README.md`，记录实验顺序、架构图、运行前提和验证结果。
- **新增 `.gitignore` 并建立根目录 git 仓库**：当前根目录没有任何版本控制，改错无法回滚（见 3.4 节）。这是改动实验代码前的安全保障。
- 为 StateGraph 路由、工具权限、资料合并和 Store 隔离建立不依赖真实模型的最小测试（可用 `create_agent` + Fake Model 方式，已在本仓库验证可行）。

### 12.5 后续学习任务

- 使用持久化 checkpoint 完成 Human-in-the-loop 的暂停、审批和恢复实验。
- 学习 MCP、RAG、Subgraph、并行和 Supervisor/Multi-Agent。
- 选择最终简历项目的真实业务场景。
- 建立评估数据、Trace、成本统计和失败分析机制。
- 在具备足够基础后，开始按组件阅读 DeerFlow 源码，并记录架构对照笔记。

### 12.6 待决策事项

- `short-memory.db`（741 KB，孤立文件，最后写入 2026-09-07）是否删除。**删除前必须得到维护者明确授权**，见 4.11 节。
