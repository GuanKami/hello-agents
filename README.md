# hello-agents

个人 AI Agent 学习实验仓库。

这个仓库用于通过一组小型、可运行、可观察的 Python 实验，逐步理解现代 AI Agent 的核心机制，并为后续学习字节跳动 DeerFlow、开发独立 Agent Demo 和完成简历项目打基础。

> 当前状态：学习阶段，尚未进入最终项目的正式实现阶段。
>
> 截至 2026 年 10 月 16 日，LangGraph 基础、Memory、Context Engineering、Middleware、HITL、MCP 与基础 Hybrid RAG 已完成相关实验。Agentic RAG 基础路径、编号检查及人工引用正负向核对方法已阶段性收口，Subgraph 基础已完成，固定并行及 reducer 已完成，Send / Map-Reduce 基础路径已手动验证，Supervisor / Multi-Agent 三条基础路径已手动验证，下一主线 Evaluation / Observability；引用来源错配、assess 解析异常和更广评测仍是已知限制。
>
> 2026 年 9 月 26 日两次北京实时天气查询均完成工具发现、调用、结果回传和模型回答；这不代表重复工具循环、其他城市或异常路径已经验证。维护者最新提供的 Hybrid RAG 输出读取 2 份 Markdown 并切分为 9 个文本块。针对 ToolMessage 问题，RRF Top-3 为 `react.md/start=363`（Dense/BM25 均第 1，0.0328）、`react.md/start=0`（均第 2，0.0323）、`stategraph.md/start=605`（Dense 第 3、BM25 未命中，0.0159）；图实际发送给 LLM 的上下文顺序与融合排名一致，答案使用了对应的 `[D1]`、`[D2]`、`[D3]`。这是一条小型语料上的正向端到端验证，不代表系统化检索质量、引用语义或库外拒答评估。此前输出里的 `It's always sunny ...` 来自固定天气示例，不作为真实天气接口的验证记录。

### RAG 基础流程

[`langgraph_rag_demo.py`](langgraph_rag_demo.py) 是固定两步 Hybrid RAG StateGraph 实验，不是 Agentic RAG：

```text
本地 Markdown -> 文档切块 -> Dense 向量索引 + BM25 关键词索引
    -> retrieve 节点：Dense Top-3 + BM25 Top-3 -> RRF Top-3
    -> 将融合排序后的 Document 写入 RAGState.documents
    -> generate 节点显式构造模型上下文并生成答案
```

- 知识库位于 [`rag_docs/`](rag_docs/)；目前包括 `react.md` 和 `stategraph.md`。
- `RecursiveCharacterTextSplitter` 使用 `chunk_size=400`、`chunk_overlap=80`，并记录 `start_index`，便于回看文本块在原文中的位置。
- 图内 `retrieve` 节点通过 `hybrid_retrieve()` 取 Dense 与 BM25 两路 Top-3，再由 RRF 排序；`generate` 节点将问题和融合后的证据片段组装为消息。文档保存在 State 中并不意味着 LangGraph 会自动把它们传给 LLM。
- 维护者提供的端到端结果读取 2 份原始文档并切成 9 个片段。针对“ReAct 中 ToolMessage 代表什么？工具返回后 Agent 如何继续？”，RRF 排名依次为 `react.md/start=363`（两路均第 1，0.0328）、`react.md/start=0`（两路均第 2，0.0323）、`stategraph.md/start=605`（Dense 第 3、BM25 未命中，0.0159）。输出显示图实际发送给 LLM 的上下文与该顺序相同，生成答案使用了对应编号。
- `[D1]`、`[D2]`、`[D3]` 是按本次 RRF 最终顺序临时分配的编号，可定位回当前上下文片段，但不是稳定文档 ID，也不自动证明引用内容直接支持结论。该运行验证的是一条正向路径；更广的问题集与引用语义评估仍可后续补充。
- 运行时会重新嵌入文档，并分别执行一次独立 Dense 诊断检索和一次图内混合检索；因此会重复计算一次查询向量并产生额外 Embedding 请求。还会调用 Chat API，可能产生费用。`InMemoryVectorStore` 仅在当前进程有效。

### BM25 与混合检索实验

[`langgraph_bm25_demo.py`](langgraph_bm25_demo.py) 和 [`langgraph_hybrid_rag_demo.py`](langgraph_hybrid_rag_demo.py) 用与向量 RAG 相同的 9 个文本块，分别观察关键词检索以及 Dense + BM25 的排名融合：

```text
langgraph_bm25_demo.py:
文本块 -> jieba 中文分词 -> BM25Okapi -> BM25 Top-3

langgraph_hybrid_rag_demo.py:
文本块 -> Dense Top-3 ─┐
文本块 -> BM25 Top-3  ─┴-> RRF -> 融合 Top-3
```

- BM25 是词项检索，不需要 Embedding；当前独立脚本只加载本地资料、分词、排序和打印结果，不调用 LLM 或 Embedding API。
- 混合检索使用 Reciprocal Rank Fusion（RRF）融合两路排名，不把向量相似度分数和 BM25 分数直接相加。脚本会打印每个融合结果在 Dense 与 BM25 中各自的名次，方便追踪排名来源。
- `langgraph_hybrid_rag_demo.py` 的独立入口用于分别查看 Dense、BM25 和 RRF；其中 `hybrid_retrieve()` 已被 `langgraph_rag_demo.py` 的 `retrieve` 节点调用，完成了固定两步 Hybrid RAG 集成。独立对比脚本不生成答案，完整生成流程由 RAG StateGraph 执行。
- 手动运行中，BM25 的 3 个库内问题找到了相关片段；混合检索的多个独立查询也观察到两路候选。集成图目前有 ToolMessage 问题的一条端到端正向结果。询问知识库未覆盖的 DeerFlow 长期记忆数据库时，BM25 和 RRF 仍会返回无关 Top-K 候选。**Top-K / RRF 只负责排序，不等于确认知识库存在答案，也不会自动拒答。**
- RAG 基础里程碑已完成；扩展改写问法、跨文档、引用语义和库外拒答评估属于后续质量改进。当前小型语料与手动样例不构成系统评测，也没有持久化向量库、相关性阈值或重排器。

### Agentic RAG 第一版：证据评估与有界重试

[`langgraph_agentic_rag_demo.py`](langgraph_agentic_rag_demo.py) 在固定两步 Hybrid RAG 前后加入规划与证据评估：

```text
START -> decide --direct--> direct_answer -> END
             └--retrieve--> retrieve -> Hybrid(Dense + BM25 + RRF) -> assess
                                                                      ├ sufficient -> grounded_generate -> END
                                                                      ├ retry -> apply_retry_query -> retrieve
                                                                      └ no_evidence -> END
```

- `RetrievalPlan` 通过结构化输出给出 `action` 和首轮 `query`。当前代码用 Python 规则识别固定集合中的独立问候/感谢并进入 `direct`；其他输入即使 planner 输出 `direct`，有效路由也固定为 `retrieve`，planner 只提供查询改写。这样可避免知识问题被 planner 误路由跳过检索，但不是通用意图分类策略。
- `retrieve` 复用 `langgraph_hybrid_rag_demo.py` 的 `hybrid_retrieve()`；`assess` 使用 `EvidenceReview` 判断当前片段是否直接支持回答，并在不足时提出改写查询。
- `route_after_assess` 由程序检查评估结果、查询是否为空或与原查询相同，以及 `MAX_RETRIEVALS=2`。这里的 2 包含首轮检索，因此最多是首轮加一次重试；达到边界后进入 `no_evidence`。`grounded_generate` 显式将问题和证据加入模型上下文。
- `direct_answer` 在图运行时绕过检索节点，不声称查过本地资料；但 `main()` 仍会在 `invoke()` 前加载语料、建立索引，所以 direct 仍有启动成本。
- 固定路由手动验收观察到：“你好。”进入 `direct`；库内 StateGraph 问题进入 `retrieve`；“你好 + StateGraph 知识问题”仍进入 `retrieve`；库外 DeerFlow 数据库问题进入 `retrieve`，经首轮和一次重试后进入 `no_evidence`。这验证的是当前固定规则覆盖的少量输入，不代表通用路由准确率或拒答准确率。旧版 planner 误选 `direct` 的问题已由固定路由策略规避。
- 生成节点输出普通文本“结论、引用、证据原文”。`inspect_citation_ids()` 只检查全文编号存在性，不检查逐条引用覆盖、摘录归属或语义支持，不拦截或修复答案。missing_citations=False 仅指匹配到编号，只有无效 D99 时也为 False。
- 第二轮 assess 解析失败后在上限处保守进入 no_evidence 已观察，这是异常兜底而非正常证据不足判断。首轮解析失败仍抛错，网络和模型调用异常未统一处理。
- 运行时会调用 Embedding API 和 Chat API；向量索引使用 `InMemoryVectorStore`，每次进程启动后重新建立，不提供持久化。

引用验收记录（2026-10-13，依据维护者提供的输出）：

| 用例 | 检查结果 | 验收边界 |
| --- | --- | --- |
| State/D1、Node/D2、Edge/D2 的正向回答 | 编号存在，摘录在所引片段内并支持结论 | 一条正向样例通过 |
| 测试文本引用 D99 | invalid_ids 包含 D99 | 无效编号检查通过 |
| Node 返回字段更新却引用有效 D3 | 编号通过；本轮 D3 不支持该结论 | 人工可识别不支持 |
| 实际 Edge 回答标 D1，摘录来自 D2 | 编号通过；人工发现来源错配 | 引用正确性仍有缺陷 |

基础验收方法完成不代表引用稳定，应保留失败样例。当前 main() 保留有效 D3 的负向调用但没有打印报告，已贴出的负向日志是此前手动验收记录。所有结束分支统一打印“Agent 最终回答”；no_evidence 是程序固定文本，其余两支由模型生成。

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
  -> MCP / Hybrid RAG / Agentic RAG / Subgraph / Multi-Agent
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
- 使用 `wrap_tool_call` 观察工具调用参数、`tool_call_id` 和 `ToolMessage`。
- 工具权限 middleware：根据运行时 `Context` 决定放行或短路工具执行。
- 工具异常 middleware：捕获 `ZeroDivisionError` 并转换为模型可读取的错误 `ToolMessage`。

### Human-in-the-loop

- 使用 `HumanInTheLoopMiddleware` 在工具执行前产生 `__interrupt__`。
- 使用 `InMemorySaver` 保存暂停时的 State 和恢复位置。
- 使用同一个 `thread_id` 和 `Command(resume=...)` 恢复执行。
- `approve` 路径已验证：工具执行并返回正常 `ToolMessage`。
- `reject` 路径已验证：工具不执行，返回拒绝 `ToolMessage`，当前提示词下模型不会重试。
- 使用 `SqliteSaver` 将 checkpoint 保存到 `hitl-checkpoint.db`。
- 使用 `start` / `resume` 两种命令，在两个独立 Python 进程中完成暂停和恢复。
- `test01` 的 approve 跨进程恢复和 `test02` 的 reject 跨进程恢复均已验证。

### MCP 与真实天气服务

- `langgraph_mcp_demo.py` 使用 LangChain 内置 `MCPAdapter`，通过 `mcpServers` 配置以 stdio 启动独立 MCP Server，并在 `async with` 生命周期内调用 `list_tools()` 发现工具，再交给 `create_agent`。
- Agent 的 `ainvoke()` 也在适配器上下文内完成，以保证模型运行期间 MCP 连接仍然有效；Agent 仍负责 ReAct 工具循环，适配器只负责 MCP 连接和工具适配。
- `mcp_server/get_weather_mcp/server.py` 使用 FastMCP 暴露 `get_weather`，并通过 HTTP 调用和风天气服务：先把中国城市名解析为经纬度，再请求当前天气。
- 共享模块 [`tools.py`](tools.py) 中仍保留一个固定文本的 `get_weather` 教学工具；本实验不会直接导入它，而是使用 MCP Server 发现到的真实天气工具。
- 维护者于 **2026 年 9 月 26 日**使用 `MCPAdapter` 运行两次真实请求，均发现并调用 `get_weather(city="北京")`，工具结果经 MCP 返回给 Agent，模型随后作答。两次结果分别包括晴间多云（26.59°C、体感 28.79°C、湿度约 59%）和小雨（18°C、体感 17.83°C、湿度约 90%）。
- 这些记录证明两次单工具实时天气请求成功，不代表多工具 ReAct 循环、其他城市、异常路径或逐日天气预报已验证。湿度结果与服务端按 0–1 比例格式化的假设相符，但不替代完整 Schema 和边界值检查。地点名“北京市北京”仍有重复，可后续修正展示。
- 此前输出里的固定“始终晴朗”内容属于旧教学工具，不是实时天气结果。

Agentic RAG 基础路径与引用验收方法已阶段性收口，Subgraph 基础已完成，固定并行与 reducer 已完成，Send / Map-Reduce 三主题基础路径已验证，Supervisor / Multi-Agent 三条基础路径已验证，后续进入 Evaluation / Observability。质量任务仍包括摘录归属/语义检查、assess 首轮解析失败及调用异常、更广评测。Middleware 重试和 HITL 扩展仍是补充项。具体状态以 [AGENTS.md](AGENTS.md) 为准。

## 实验文件

### Subgraph 基础实验

`langgraph_subgraph_demo.py` 复用已编译的 Agentic RAG 工作流，父图结构为：

```text
START -> prepare -> rag（子图）-> format_result -> END
```

prepare 将 user_input 转为 question；父子图共同声明 question 和 answer，
直接通过 `add_node("rag", rag_graph)` 接入；format_result 读取 answer 并整理 final_answer。
子图的 documents、retrieval_count 等私有字段不自动成为父图输出字段。

2026-10-13，维护者确认包装节点调用与直接注册两种方式已手动运行，知识问题和问候均得到父图最终回答；当前文件只保留直接注册版本。Subgraph 基础完成，固定并行及 reducer 已完成，Send / Map-Reduce 基础路径已手动验证，Supervisor / Multi-Agent 三条基础路径已手动验证，下一主线 Evaluation / Observability。

此实验是串行图组合，尚未实现并行、Multi-Agent、父子图 checkpoint 或嵌套流式输出。main 在图运行前建立向量索引，问候也会产生 Embedding 初始化请求；真实运行会访问 Embedding / Chat API。子图沿用 RAG 引用来源错配和评估解析异常的已知限制。

### 固定并行与 reducer

`langgraph_parallel_demo.py` 将概念解释与代码示例分为两个独立模型节点：

```text
START -> prepare -> explain_concept / give_example -> combine -> END
```

两条分支各调用一次 Chat 模型，写入同一个 results 列表；operator.add reducer 拼接更新，
汇合边等待两条分支完成。combine 按 kind 读取结果，以 Python 拼接答案，不额外请求模型。
维护者于 2026-10-14 提供当前版本结果：分支耗时 53.83s / 55.85s，总耗时约 55.85s，
合并列表保留 concept 和 example。固定并行及 reducer 已手动验证；不是系统性能基准。
模型生成内容未经正确性验证，示例不执行。未覆盖单分支失败、限流或超时；未使用 Send、
checkpoint、并行子图或 Multi-Agent。动态分发已在独立 Map-Reduce 实验中验证。

运行当前实验（两次 Chat 请求可能产生费用，无 Embedding 请求）：

```powershell
.\.venv\Scripts\python.exe langgraph_parallel_demo.py
```

### Send 动态分发与 Map-Reduce

[`langgraph_map_reduce_demo.py`](langgraph_map_reduce_demo.py) 根据本次输入决定任务数量：

```text
START -> prepare -> Send(explain_topic) × N -> combine -> END
```

prepare 清理空白、过滤空主题并按首次出现顺序去重；每个 Send 只传入自己的 topic。
图里只注册一个 explain_topic 节点，每个任务调用一次 Chat 模型，返回单项 results 列表。
operator.add reducer 拼接这些更新；combine 用 topic 对应结果，按输入主题顺序拼接答案，
不依赖任务完成顺序，也不额外调用模型。全为空时抛 ValueError。

2026-10-16，维护者提供的三主题运行中，State、Node、Edge 均开始后，按 Node、Edge、State
顺序完成；combine 收到 3 项，最终按 State、Node、Edge 展示。动态分发与汇总的基础路径
已手动验证；其他任务数量、空输入、去重和失败路径尚未运行验收。

生成解释未核验准确性，“不超过150字”是提示词约束。当前没有检索证据、显式并发限制、
失败降级、重试或 checkpoint。每个去重主题请求一次 Chat API，可能产生费用；不调用
Embedding。Supervisor / Multi-Agent 基础已验证，下一主线是 Evaluation / Observability，并行子图作为可选补充。

```powershell
.\.venv\Scripts\python.exe langgraph_map_reduce_demo.py
```

### Supervisor / Multi-Agent：专家作为工具

`langgraph_supervisor_demo.py` 由主 Agent 选择专家，并利用专家结果组织回答：

```text
用户 -> Supervisor -> ask_knowledge_agent -> 现有 Agentic RAG -> answer
                  -> ask_math_agent -> 数学 Agent -> divide -> 计算回答
                 <- 两个专家的结果作为 ToolMessage 返回
     <- Supervisor 最终回答
```

专家仅接收本次子任务，返回答案文本；父会话不会自动传入专家，专家的完整 State 和
内部工具消息也不会自动返回父会话。多个角色共用模型客户端，分别配置提示词与工具。
create_agent 在内部组织模型调用、工具执行和回到模型的循环。

2026-10-16，维护者提供的三条真实运行均符合预期：

| 输入场景 | 实际行为 |
| --- | --- |
| 解释 StateGraph 中 Node 的职责 | 只调用知识专家，经 RAG 返回带引用答案 |
| 计算 2246 除以 10 | 只调用数学专家，执行 divide，返回 224.6 |
| 同时解释 Node 并计算除法 | 同轮调用两个专家，接收结果后汇总；日志观察到执行重叠 |

这是三条正常路径的手动验证，不代表通用路由准确率或异常恢复已验证。知识专家的 D#
仍属于单次检索上下文，返回文本没有独立来源映射；RAG 的引用错配和 assess 解析异常
仍可能发生。没有配置 checkpoint、统一专家异常转换、显式超时或总 API 请求预算；
recursion_limit=20 是父图步数约束。数学工具零除抛 ValueError，此路径未验收。

main 默认运行混合问题；修改 question 可复跑单任务。启动均会建立内存向量索引，
因此纯计算或寒暄也有 Embedding 初始化请求，运行还会产生多次 Chat 请求。

```powershell
.\.venv\Scripts\python.exe langgraph_supervisor_demo.py
```

下一阶段为 Evaluation / Observability：先比较固定用例的预期/实际专家选择，记录异常、
最终答案和总耗时。该评估脚本尚未实现，证据支持度需要独立核对。

### 实验目录

| 文件 | 学习内容 | 当前状态 |
| --- | --- | --- |
| `langgraph_supervisor_demo.py` | 专家作为工具、子任务上下文、结果回传与主 Agent 汇总 | 仅知识、仅数学、混合任务三条正常路径已手动验证；下一步评估与可观测性 |
| [`langgraph_map_reduce_demo.py`](langgraph_map_reduce_demo.py) | Send 动态分发、局部任务输入、列表 reducer 与按主题汇总 | 三主题正常路径已手动验证；更多规模和异常路径未验证 |
| [`langgraph_parallel_demo.py`](langgraph_parallel_demo.py) | 固定分叉、等待汇合、operator.add 合并共享 results | 已由维护者手动验证；保留固定分支基线，异常路径未验证 |
| [`langgraph_rag_demo.py`](langgraph_rag_demo.py) | 固定两步 Hybrid RAG：Dense + BM25 + RRF、State 传递和带引用生成 | 已完成一次端到端正向运行，图内 RRF 名次与模型上下文/D# 顺序一致；更广评测与库外拒答待后续补充 |
| [`langgraph_bm25_demo.py`](langgraph_bm25_demo.py) | jieba 中文分词与 BM25 关键词 Top-K 检索 | 独立单路基线；3 个库内查询命中相关片段，库外问题仍返回候选 |
| [`langgraph_hybrid_rag_demo.py`](langgraph_hybrid_rag_demo.py) | Dense + BM25 + RRF 排名融合与可复用 helper | 独立对比入口已验证；`hybrid_retrieve()` 已集成到 RAG StateGraph |
| [`langgraph_agentic_rag_demo.py`](langgraph_agentic_rag_demo.py) | 固定路由、证据评估、有界重试、编号诊断与人工核对 | 基础验收方法收口；来源错配与解析异常仍保留，Subgraph 基础已完成，固定并行及 reducer 已完成，Send / Map-Reduce 基础路径已手动验证，Supervisor / Multi-Agent 三条基础路径已手动验证，下一主线 Evaluation / Observability |
| [`tools.py`](tools.py) | 共享工具、权限控制、用户资料读写 | 已使用 |
| [`langgraph_subgraph_demo.py`](langgraph_subgraph_demo.py) | 父子图组合、共享 question / answer、结果整理 | 两种接入方式由维护者手动验证；当前直接注册版，后续动态分发见 Map-Reduce 实验 |
| [`langgraph_react.py`](langgraph_react.py) | 高层 `create_agent` Agent API | 学习实验，存在待修正缺陷 |
| [`langgraph_state_react.py`](langgraph_state_react.py) | 显式 StateGraph ReAct 和 ToolNode | 已完成基础实验 |
| [`embedding_test.py`](embedding_test.py) | Embedding、Store 和语义检索 | 已完成最小实验 |
| [`langgraph_context_demo.py`](langgraph_context_demo.py) | Runtime + Store 构造 Model Context | 已完成 |
| [`langgraph_state_context_demo.py`](langgraph_state_context_demo.py) | State 驱动 Model Context | 已完成 |
| [`langgraph_middleware_dynamic_prompt_demo.py`](langgraph_middleware_dynamic_prompt_demo.py) | `@dynamic_prompt` | 已完成 |
| [`langgraph_middleware_hooks_demo.py`](langgraph_middleware_hooks_demo.py) | `before_model` / `after_model` 写入 State，并复现动态换模型 | 已完成真实模型验证 |
| [`langgraph_middleware_wrap_model_call_demo.py`](langgraph_middleware_wrap_model_call_demo.py) | wrapper 观察、模型调用链和本地短路 | 换模型与短路已验证，重试未做 |
| [`langgraph_middleware_wrap_tool_call_demo.py`](langgraph_middleware_wrap_tool_call_demo.py) | 工具调用前后观察、`handler(request)` 和 `ToolMessage` | 已完成真实模型验证 |
| [`langgraph_middleware_tool_guard_demo.py`](langgraph_middleware_tool_guard_demo.py) | 基于 `Context.authority` 的工具权限放行和短路 | 已完成真实模型验证 |
| [`langgraph_middleware_tool_error_demo.py`](langgraph_middleware_tool_error_demo.py) | 捕获 `ZeroDivisionError` 并转换为错误 `ToolMessage` | 已完成真实模型验证 |
| [`langgraph_human_in_the_loop_demo.py`](langgraph_human_in_the_loop_demo.py) | `interrupt`、审批、checkpoint 和 `Command(resume=...)` | approve/reject 已验证，持久化恢复未做 |
| [`langgraph_human_in_the_loop_sqlite_demo.py`](langgraph_human_in_the_loop_sqlite_demo.py) | `SqliteSaver`、`start/resume` 和跨进程 checkpoint 恢复 | approve/reject 已验证，负向 thread_id/edit/respond 未做 |
| [`langgraph_mcp_demo.py`](langgraph_mcp_demo.py) | 内置 `MCPAdapter`、stdio 工具发现与 Agent 调用 | 已验证两次真实天气调用、结果回传和回答（2026-09-26）；多工具循环及异常路径未验证 |
| [`mcp_server/get_weather_mcp/server.py`](mcp_server/get_weather_mcp/server.py) | FastMCP 天气 Server，通过 HTTP 查询真实天气 | 已接入和风天气 API；两次北京实时天气请求成功（2026-09-26） |

根目录的 `dive-into-langgraph/` 是独立的课程源码仓库，必须保留；`.agents/skills/dive-into-langgraph/` 是本地学习 Skill，也必须保留。

## 目录和职责

当前项目保持平铺式学习脚本结构：

```text
hello-agents/
├── tools.py                                      # 共享工具、Context 和用户资料读写
├── langgraph_react.py                            # 高层 Agent API 实验
├── langgraph_state_react.py                      # 显式 StateGraph ReAct
├── embedding_test.py                             # Embedding / Store 语义检索
├── langgraph_context_demo.py                    # Runtime + Store -> Model Context
├── langgraph_state_context_demo.py              # State -> Model Context
├── langgraph_middleware_dynamic_prompt_demo.py  # @dynamic_prompt
├── langgraph_middleware_hooks_demo.py           # before_model / after_model
├── langgraph_middleware_wrap_model_call_demo.py # wrap_model_call
├── langgraph_middleware_wrap_tool_call_demo.py  # wrap_tool_call 观察
├── langgraph_middleware_tool_guard_demo.py      # 工具权限闸门
├── langgraph_middleware_tool_error_demo.py      # 工具异常转换
├── langgraph_human_in_the_loop_demo.py          # Human-in-the-loop 审批
├── langgraph_human_in_the_loop_sqlite_demo.py   # SqliteSaver 跨进程 HITL
├── langgraph_mcp_demo.py                        # MCPAdapter + Agent
├── langgraph_rag_demo.py                        # 固定两步 Hybrid RAG 图（Dense + BM25 + RRF）
├── langgraph_bm25_demo.py                       # 中文 BM25 关键词检索
├── langgraph_hybrid_rag_demo.py                 # Dense + BM25 + RRF 对比和检索 helper
├── langgraph_agentic_rag_demo.py                # Agentic RAG：规划、证据评估、有界重试与拒答
├── langgraph_subgraph_demo.py                  # 父图直接注册 RAG 子图，共享输入输出字段
├── langgraph_parallel_demo.py                  # 固定并行与 results 列表 reducer
├── langgraph_map_reduce_demo.py                # Send 动态分发与 Map-Reduce
├── langgraph_supervisor_demo.py                # Supervisor 委派知识与数学专家
├── rag_docs/
│   ├── react.md                                  # RAG 知识库样例
│   └── stategraph.md                             # RAG 知识库样例
├── mcp_server/
│   └── get_weather_mcp/server.py                # FastMCP + 真实天气 API
├── requirements.txt                              # Python 依赖声明
├── AGENTS.md                                    # 长期协作规范和真实进度
├── hitl-checkpoint.db                            # SQLite checkpoint 运行产物，已被忽略
├── .agents/                                     # 本地 Skill 配置
└── dive-into-langgraph/                         # 课程资料，独立仓库
```

当前根目录没有正式测试目录、CI 配置、构建配置、部署配置或统一应用入口。每个脚本主要服务于一个学习目标，不应仅为了减少文件数量而过早抽象。

## 技术栈

- Python
- LangChain
- LangChain OpenAI
- LangGraph
- Pydantic
- python-dotenv
- FastMCP
- `langchain[mcp]`（内置 `MCPAdapter`，当前 Beta）
- `langchain-text-splitters`（RAG 文档切分）
- `rank-bm25`（BM25 关键词检索）
- `jieba`（BM25 中文分词）
- httpx
- 和风天气 HTTP API（运行时外部服务）
- OpenAI 兼容接口
- SerpApi（共享搜索工具使用）
- IPython（消息展示和 Notebook 支持）

根目录 `requirements.txt` 是当前明确声明的依赖列表，包含 `langchain[mcp]>=1.4.2`、FastMCP `>=4.0.10,<5`、LangGraph、Pydantic 和 `httpx` 等实验依赖；MCP 集成由 LangChain 内置适配器提供，不再单独依赖 `langchain-mcp-adapters`。和风天气是运行时外部服务，不是可通过 pip 安装的包。

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
QWEATHER_API_HOST
QWEATHER_API_KEY
```

`QWEATHER_API_HOST` 填天气服务的 Host（不要包含密钥）；当前服务端会自行补上 `https://`。
具体模型、接口地址和密钥由本地环境决定，README 不记录真实配置。

## 安装和运行

仓库目前没有官方安装脚本、启动脚本、测试脚本或构建脚本。可以根据 `requirements.txt` 安装依赖，具体实验使用虚拟环境中的 Python 逐个运行。

在 Windows PowerShell 中，当前常用入口如下：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe langgraph_state_react.py
.\.venv\Scripts\python.exe embedding_test.py
.\.venv\Scripts\python.exe langgraph_rag_demo.py
.\.venv\Scripts\python.exe langgraph_bm25_demo.py
.\.venv\Scripts\python.exe langgraph_hybrid_rag_demo.py
.\.venv\Scripts\python.exe langgraph_agentic_rag_demo.py
.\.venv\Scripts\python.exe langgraph_subgraph_demo.py
.\.venv\Scripts\python.exe langgraph_supervisor_demo.py
.\.venv\Scripts\python.exe langgraph_context_demo.py
.\.venv\Scripts\python.exe langgraph_state_context_demo.py
.\.venv\Scripts\python.exe langgraph_middleware_dynamic_prompt_demo.py
.\.venv\Scripts\python.exe langgraph_middleware_hooks_demo.py
.\.venv\Scripts\python.exe langgraph_middleware_wrap_model_call_demo.py
.\.venv\Scripts\python.exe langgraph_middleware_wrap_tool_call_demo.py
.\.venv\Scripts\python.exe langgraph_middleware_tool_guard_demo.py
.\.venv\Scripts\python.exe langgraph_middleware_tool_error_demo.py
.\.venv\Scripts\python.exe langgraph_human_in_the_loop_demo.py
.\.venv\Scripts\python.exe langgraph_mcp_demo.py

# SQLite HITL：第一次命令创建中断，第二次命令在新的进程中恢复
.\.venv\Scripts\python.exe langgraph_human_in_the_loop_sqlite_demo.py start test01
.\.venv\Scripts\python.exe langgraph_human_in_the_loop_sqlite_demo.py resume test01 approve
```

语法检查可以使用：

```powershell
.\.venv\Scripts\python.exe -m py_compile tools.py langgraph_react.py langgraph_state_react.py embedding_test.py langgraph_rag_demo.py langgraph_bm25_demo.py langgraph_hybrid_rag_demo.py langgraph_agentic_rag_demo.py langgraph_subgraph_demo.py langgraph_parallel_demo.py langgraph_map_reduce_demo.py langgraph_supervisor_demo.py langgraph_context_demo.py langgraph_state_context_demo.py langgraph_middleware_dynamic_prompt_demo.py langgraph_middleware_hooks_demo.py langgraph_middleware_wrap_model_call_demo.py langgraph_middleware_wrap_tool_call_demo.py langgraph_middleware_tool_guard_demo.py langgraph_middleware_tool_error_demo.py langgraph_human_in_the_loop_demo.py langgraph_human_in_the_loop_sqlite_demo.py langgraph_mcp_demo.py mcp_server/get_weather_mcp/server.py
```

MCP 命令会启动本地 stdio Server，并请求真实天气 API 和 LLM；可能产生外部请求或费用。运行前需要在根目录 `.env` 配置 `LLM_MODEL_ID`、`LLM_API_KEY`、`LLM_BASE_URL`、`QWEATHER_API_HOST` 和 `QWEATHER_API_KEY`。模型、Embedding、搜索和天气工具都可能访问外部 API；不要把 `.env` 提交到 Git，也不要在日志中打印密钥。

没有测试框架时，脚本运行结果属于手动验证，不能等同于自动化测试。进行低成本静态检查时，优先使用 AST 解析或语法检查，不要为了导入模块而触发模块顶层的模型/Embedding 请求。

部分实验使用 `InMemoryStore` 或 `InMemorySaver`，数据只在当前 Python 进程内有效；
`thread_id` 只能在相同 checkpoint 后端仍可访问时恢复对应会话。SQLite HITL 实验使用
`SqliteSaver` 将 checkpoint 写入根目录的 `hitl-checkpoint.db`，该文件是运行产物，
不是长期用户资料，也不会提交到 Git。

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
  -> wrap_model_call 重试 / fallback（可选补充）
  -> Human-in-the-loop edit/respond 与负向 thread_id 验证（可选补充）
  -> MCP 天气 Agent（已验证两次真实天气工具调用和结果回传；其他路径未验证）
  -> 固定两步 Hybrid RAG 已接入并完成一次端到端正向运行，RRF 名次、模型上下文和 D# 映射一致
  -> Agentic RAG：基础路径、编号检查与人工正负向核对方法收口；引用错配与 assess 异常为后续质量任务
  -> 固定两步 RAG 的改写问法、跨文档、引用语义及库外拒答扩展评测（可选质量改进）
  -> Subgraph 包装调用与直接注册 [已手动验证，当前保留直接注册版]
  -> 固定并行节点与等待汇合、operator.add reducer [已手动验证，2026-10-14]
  -> Send / Map-Reduce [三主题正常路径已手动验证，2026-10-16]
  -> 并行子图 [可选补充，尚未学习]
  -> Supervisor / Multi-Agent [三条基础路径已手动验证，2026-10-16]
  -> Evaluation / Observability [下一主线，尚未实现]
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
