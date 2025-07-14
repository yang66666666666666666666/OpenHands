# OpenHands Memory Module Architecture

## Overview

The Memory module in OpenHands is responsible for managing the agent's memory and context during conversations. It has two primary responsibilities:

1. **Knowledge Retrieval**: Providing access to repository information, runtime context, and domain-specific knowledge through microagents
2. **Context Management**: Condensing conversation history to maintain context while staying within token limits

## Technical Stack

- **Python 3.12+**: Core programming language
- **Pydantic**: For data modeling and validation
- **Asyncio**: For asynchronous operations
- **Event-driven Architecture**: Using EventStream for communication between components
- **Factory Pattern**: For creating condenser instances from configurations
- **Strategy Pattern**: For implementing different condensation strategies

## Core Components

### 1. Memory

The `Memory` class is the main entry point for knowledge retrieval. It:

- Listens to the EventStream for RecallAction events
- Loads and manages microagents from different sources
- Provides workspace context and microagent knowledge when requested
- Stores repository and runtime information

#### Microagent Types

- **RepoMicroagents**: Always active, provide repository-specific context
- **KnowledgeMicroagents**: Triggered by specific keywords, provide domain-specific knowledge

#### Microagent Sources

- **Global Microagents**: Installed with OpenHands, available to all users
- **User Microagents**: Stored in ~/.openhands/microagents/, specific to the user
- **Workspace Microagents**: Found in the user's cloned repository (.openhands/microagents/)

### 2. Condenser

The `Condenser` module is responsible for managing conversation history size by condensing events when necessary. It:

- Reduces the number of events that need to be included in the LLM context
- Summarizes or filters events to stay within token limits
- Maintains important context while removing less relevant information

#### Condenser Strategies

The module implements several condensation strategies:

- **LLMSummarizingCondenser**: Uses the LLM to generate summaries of forgotten events
- **AmortizedForgettingCondenser**: Gradually forgets older events
- **ConversationWindowCondenser**: Keeps a fixed window of recent events
- **BrowserOutputCondenser**: Specializes in condensing browser output
- **ObservationMaskingCondenser**: Masks certain parts of observations
- **NoOpCondenser**: Passes through events without condensation
- **RecentEventsCondenser**: Keeps only the most recent events
- **StructuredSummaryCondenser**: Creates structured summaries of events

### 3. View

The `View` class represents a filtered and processed subset of the event history that is ready to be sent to the LLM. It:

- Provides list-like access to the filtered events
- Handles the semantics of condensation events
- Ensures forgotten events are properly excluded
- Inserts summaries at the appropriate positions

## Data Flow

1. **Knowledge Retrieval Flow**:
   - Agent sends a RecallAction to the EventStream
   - Memory receives the action and processes it based on its type
   - For WORKSPACE_CONTEXT recalls, Memory collects repository info, runtime info, and microagent knowledge
   - For KNOWLEDGE recalls, Memory finds matching microagent knowledge based on triggers
   - Memory creates a RecallObservation with the retrieved information
   - The observation is added to the EventStream for the agent to process

2. **Condensation Flow**:
   - Agent requests the condensed history from the current State
   - State delegates to the configured Condenser
   - Condenser checks if condensation is needed based on its strategy
   - If needed, Condenser creates a Condensation with a summary of forgotten events
   - The Condensation is added to the event history
   - On the next request, View.from_events processes the history to exclude forgotten events and insert summaries

## Key Interfaces

### Memory Interface

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

### Condenser Interface

```python
class Condenser(ABC):
    @abstractmethod
    def condense(self, view: View) -> View | Condensation
    def condensed_history(self, state: State) -> View | Condensation
    @classmethod
    def from_config(cls, config: CondenserConfig) -> Condenser
```

### View Interface

```python
class View(BaseModel):
    events: list[Event]
    unhandled_condensation_request: bool

    def __getitem__(self, key: int | slice) -> Event | list[Event]
    @staticmethod
    def from_events(events: list[Event]) -> View
```

## Extension Points

The Memory module is designed to be extensible in several ways:

1. **New Condenser Strategies**: Create a new condenser class that inherits from Condenser or RollingCondenser and implements the required methods
2. **Custom Microagents**: Add new microagents to any of the three sources (global, user, workspace)
3. **MCP Tools**: Repository microagents can define MCP tools that extend the agent's capabilities

## Integration with Other Components

- **EventStream**: Memory and Condenser communicate with other components through the EventStream
- **Agent**: Uses the condensed history to make decisions and generate responses
- **State**: Stores the event history and metadata for condensation
- **LLM**: Used by some condensers to generate summaries of forgotten events

## Performance Considerations

- **Token Management**: Condensers help manage token usage by reducing the size of the context
- **Metadata Tracking**: Condensers track metadata about condensation operations for debugging
- **Efficient Filtering**: View efficiently filters and processes events to avoid unnecessary operations
