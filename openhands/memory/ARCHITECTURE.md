# OpenHands 记忆模块架构

## 概述

OpenHands 中的记忆模块负责在对话过程中管理代理的记忆和上下文。它有两个主要职责：

1. **知识检索**：通过微代理提供对仓库信息、运行时上下文和领域特定知识的访问
2. **上下文管理**：压缩对话历史以维持上下文，同时保持在令牌限制内

## 技术栈

- **Python 3.12+**：核心编程语言
- **Pydantic**：用于数据建模和验证
- **Asyncio**：用于异步操作
- **事件驱动架构**：使用 EventStream 进行组件间通信
- **工厂模式**：从配置创建压缩器实例
- **策略模式**：实现不同的压缩策略

## 核心组件

### 1. 记忆（Memory）

`Memory` 类是知识检索的主要入口点。它：

- 监听 EventStream 中的 RecallAction 事件
- 从不同来源加载和管理微代理
- 在请求时提供工作区上下文和微代理知识
- 存储仓库和运行时信息

#### 微代理类型

- **仓库微代理（RepoMicroagents）**：始终激活，提供仓库特定上下文
- **知识微代理（KnowledgeMicroagents）**：由特定关键词触发，提供领域特定知识

#### 微代理来源

- **全局微代理**：随 OpenHands 安装，所有用户可用
- **用户微代理**：存储在 ~/.openhands/microagents/，特定于用户
- **工作区微代理**：在用户克隆的仓库中找到（.openhands/microagents/）

### 2. 压缩器（Condenser）

`Condenser` 模块负责在必要时通过压缩事件来管理对话历史大小。它：

- 减少需要包含在 LLM 上下文中的事件数量
- 总结或过滤事件以保持在令牌限制内
- 在移除不太相关的信息的同时保持重要上下文

#### 压缩策略

该模块实现了几种压缩策略：

- **LLM 总结压缩器**：使用 LLM 生成被遗忘事件的摘要
- **分摊遗忘压缩器**：逐渐遗忘较旧的事件
- **对话窗口压缩器**：保持最近事件的固定窗口
- **浏览器输出压缩器**：专门用于压缩浏览器输出
- **观察掩蔽压缩器**：掩蔽观察的某些部分
- **无操作压缩器**：不进行压缩，直接传递事件
- **最近事件压缩器**：只保留最近的事件
- **结构化摘要压缩器**：创建事件的结构化摘要

### 3. 视图（View）

`View` 类表示事件历史的经过过滤和处理的子集，可以发送给 LLM。它：

- 提供类似列表的访问过滤后的事件
- 处理压缩事件的语义
- 确保被遗忘的事件被正确排除
- 在适当的位置插入摘要

## 数据流

1. **知识检索流程**:
   - 代理向 EventStream 发送 RecallAction
   - Memory 接收动作并根据其类型进行处理
   - 对于 WORKSPACE_CONTEXT 检索，Memory 收集仓库信息、运行时信息和微代理知识
   - 对于 KNOWLEDGE 检索，Memory 根据触发器查找匹配的微代理知识
   - Memory 创建包含检索信息的 RecallObservation
   - 观察结果被添加到 EventStream 供代理处理

2. **压缩流程**:
   - 代理从当前 State 请求压缩历史
   - State 委托给配置的 Condenser
   - Condenser 根据其策略检查是否需要压缩
   - 如果需要，Condenser 创建包含被遗忘事件摘要的 Condensation
   - Condensation 被添加到事件历史中
   - 在下一次请求时，View.from_events 处理历史以排除被遗忘的事件并插入摘要

## 关键接口

### 记忆接口

```python
class Memory:
    def __init__(self, event_stream: EventStream, sid: str, status_callback: Callable | None = None)
    def on_event(self, event: Event)
    def load_user_workspace_microagents(self, user_microagents: list[BaseMicroagent]) -> None
    def get_microagent_mcp_tools(self) -> list[MCPConfig]
    def set_repository_info(self, repo_name: str, repo_directory: str) -> None
    def set_runtime_info(self, runtime: Runtime, custom_secrets_descriptions: dict[str, str]) -> None
    def set_conversation_instructions(self, conversation_instructions: str | None) -> None
```

### 压缩器接口

```python
class Condenser(ABC):
    @abstractmethod
    def condense(self, view: View) -> View | Condensation
    def condensed_history(self, state: State) -> View | Condensation
    @classmethod
    def from_config(cls, config: CondenserConfig) -> Condenser
```

### 视图接口

```python
class View(BaseModel):
    events: list[Event]
    unhandled_condensation_request: bool

    def __getitem__(self, key: int | slice) -> Event | list[Event]
    @staticmethod
    def from_events(events: list[Event]) -> View
```

## 扩展点

记忆模块设计为可以通过多种方式扩展：

1. **新的压缩策略**：创建继承自 Condenser 或 RollingCondenser 并实现所需方法的新压缩器类
2. **自定义微代理**：向任何三个来源（全局、用户、工作区）添加新的微代理
3. **MCP 工具**：仓库微代理可以定义扩展代理能力的 MCP 工具

## 与其他组件的集成

- **EventStream**：Memory 和 Condenser 通过 EventStream 与其他组件通信
- **Agent**：使用压缩历史做出决策并生成响应
- **State**：存储事件历史和压缩元数据
- **LLM**：被某些压缩器用来生成被遗忘事件的摘要

## 性能考虑

- **令牌管理**：压缩器通过减小上下文大小来帮助管理令牌使用
- **元数据跟踪**：压缩器跟踪有关压缩操作的元数据以便调试
- **高效过滤**：View 高效地过滤和处理事件以避免不必要的操作
