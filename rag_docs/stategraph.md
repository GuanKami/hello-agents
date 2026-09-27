# StateGraph：用状态和节点组织工作流

StateGraph 是 LangGraph 中用于定义有状态工作流的核心抽象。它不是一张只用于展示的流程图，而是一个会按照节点和边真正执行任务的运行时图。图中的节点负责处理业务，边负责决定下一步流向，State 负责在节点之间传递数据。

## 三个核心概念

### State

State 是工作流的共享状态。它可以保存消息列表、用户问题、检索到的文档、任务类型和最终答案等数据。例如，RAG 工作流可以定义 `question`、`documents` 和 `answer` 字段；检索节点写入 `documents`，生成节点读取 `question` 和 `documents`，最后写入 `answer`。

### Node

Node 是一个执行具体工作的函数。常见节点包括文档加载节点、检索节点、模型节点和工具节点。节点通常读取当前 State，并返回需要合并回 State 的字段更新，而不是直接修改整个工作流对象。

### Edge

Edge 描述节点之间的执行关系。普通边表示固定跳转，例如 `START -> retrieve -> generate -> END`。条件边会根据当前 State 或模型输出选择不同路径，例如模型产生 tool call 时进入 tools 节点，没有 tool call 时直接结束。

## 用 StateGraph 表示 ReAct

一个显式的 ReAct StateGraph 通常包含模型节点和工具节点：

```text
START -> model
          |
          +-- 有 tool_calls -> tools -> model
          |
          +-- 没有 tool_calls -> END
```

模型节点负责调用绑定了工具的聊天模型。工具节点负责执行模型提出的工具调用，并把结果写回消息状态。条件路由读取模型返回的 `tool_calls`：有工具调用就继续循环，没有工具调用就返回最终答案。

## StateGraph 与流程图展示的区别

`graph.get_graph().draw_mermaid()` 只会生成当前图结构的 Mermaid 描述，便于查看静态拓扑；它本身不会替你处理业务。真正运行工作流的是编译后的图对象，例如通过 `graph.invoke(...)` 或 `graph.stream(...)` 传入输入后，LangGraph 才会依次执行节点、合并 State，并沿着边继续路由。

## StateGraph 对 RAG 的作用

在最小的两步 RAG 中，可以把流程拆成：

```text
START -> retrieve -> generate -> END
```

`retrieve` 节点根据用户问题从向量存储中召回相关文本片段，并把结果写入 `documents`。`generate` 节点把用户问题和 `documents` 拼接成模型上下文，再调用 LLM 生成带来源的回答。这样可以分别观察“检索是否正确”和“模型是否忠实使用检索结果”，比把所有逻辑放进一个函数更容易调试。

StateGraph 的优势是流程可观察、状态可检查、路由可控制；代价是需要显式设计 State schema、节点职责和边。对于简单的一问一答，直接调用模型可能更简单；对于包含工具、检索、审批或多阶段处理的 Agent，显式工作流通常更容易维护。
