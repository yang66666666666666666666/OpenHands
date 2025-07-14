"""
OpenHands 记忆模块

本模块提供 OpenHands 代理系统的核心记忆功能。
它处理信息检索、微代理知识管理以及代理对话的上下文管理。

技术栈:
- Python 3.12+
- Pydantic 用于数据建模
- Asyncio 用于异步操作
- 基于 EventStream 的事件驱动架构
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import openhands
from openhands.core.config.mcp_config import MCPConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.agent import RecallAction
from openhands.events.event import Event, EventSource, RecallType
from openhands.events.observation.agent import (
    MicroagentKnowledge,
    RecallObservation,
)
from openhands.events.observation.empty import NullObservation
from openhands.events.stream import EventStream, EventStreamSubscriber
from openhands.microagent import (
    BaseMicroagent,
    KnowledgeMicroagent,
    RepoMicroagent,
    load_microagents_from_dir,
)
from openhands.runtime.base import Runtime
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.utils.prompt import (
    ConversationInstructions,
    RepositoryInfo,
    RuntimeInfo,
)

GLOBAL_MICROAGENTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(openhands.__file__)),
    'microagents',
)

USER_MICROAGENTS_DIR = Path.home() / '.openhands' / 'microagents'


class Memory:
    """
    Memory 是一个监听 EventStream 中信息检索动作（RecallAction）并发布包含内容的观察结果
    （如 RecallObservation）的组件。

    Memory 组件作为代理的知识检索系统，提供：
    1. 通过 RepoMicroagents 访问仓库特定信息
    2. 通过 KnowledgeMicroagents 提供领域特定知识
    3. 关于运行时环境的上下文信息
    4. 对话特定指令

    它基于事件驱动模型运行，通过发布包含请求信息的 RecallObservation 事件来响应 RecallAction 事件。
    """

    # 唯一会话标识符
    sid: str
    # 用于与其他组件通信的事件流
    event_stream: EventStream
    # 用于报告状态更新的回调函数
    status_callback: Callable | None
    # 用于处理异步操作的 Asyncio 事件循环
    loop: asyncio.AbstractEventLoop | None
    # 仓库特定微代理（始终激活）
    repo_microagents: dict[str, RepoMicroagent]
    # 知识微代理（由特定关键词触发）
    knowledge_microagents: dict[str, KnowledgeMicroagent]

    def __init__(
        self,
        event_stream: EventStream,
        sid: str,
        status_callback: Callable | None = None,
    ):
        """
        初始化 Memory 组件。

        参数:
            event_stream: 用于接收和发布事件的事件流
            sid: 此记忆实例的会话标识符
            status_callback: 用于报告状态更新的可选回调函数
        """
        # 初始化核心属性
        self.event_stream = event_stream
        self.sid = sid if sid else str(uuid.uuid4())
        self.status_callback = status_callback
        self.loop = None

        # 订阅事件流以接收事件
        self.event_stream.subscribe(
            EventStreamSubscriber.MEMORY,
            self.on_event,
            self.sid,
        )

        # 初始化存储微代理的字典
        self.repo_microagents = {}
        self.knowledge_microagents = {}

        # 初始化上下文信息容器
        self.repository_info: RepositoryInfo | None = None
        self.runtime_info: RuntimeInfo | None = None
        self.conversation_instructions: ConversationInstructions | None = None

        # 从 OpenHands 安装目录加载全局微代理
        # 这些是所有用户都可用的公共微代理
        self._load_global_microagents()

        # 从用户的主目录加载用户特定的微代理
        # 这些是用户创建的自定义微代理
        self._load_user_microagents()

    def on_event(self, event: Event):
        """
        处理来自事件流的事件。

        这是委托给异步处理程序的同步入口点。

        参数:
            event: 要处理的事件
        """
        asyncio.get_event_loop().run_until_complete(self._on_event(event))

    async def _on_event(self, event: Event):
        """
        Handle an event from the event stream asynchronously.

        This method processes RecallAction events and generates appropriate responses:
        1. For WORKSPACE_CONTEXT recalls: Provides repository, runtime, and microagent information
        2. For KNOWLEDGE recalls: Provides triggered microagent knowledge

        Args:
            event: The event to handle
        """
        try:
            if isinstance(event, RecallAction):
                # WORKSPACE_CONTEXT recall - typically triggered on first user message
                # Provides comprehensive context about the workspace environment
                if (
                    event.source == EventSource.USER
                    and event.recall_type == RecallType.WORKSPACE_CONTEXT
                ):
                    logger.debug('Workspace context recall')
                    workspace_obs: RecallObservation | NullObservation | None = None

                    # Generate the workspace context observation
                    workspace_obs = self._on_workspace_context_recall(event)
                    if workspace_obs is None:
                        workspace_obs = NullObservation(content='')

                    # Link this observation to the triggering action
                    # This releases the execution flow from waiting for the retrieval to complete
                    workspace_obs._cause = event.id  # type: ignore[union-attr]

                    # Add the observation to the event stream
                    self.event_stream.add_event(workspace_obs, EventSource.ENVIRONMENT)
                    return

                # KNOWLEDGE recall - triggered by specific keywords in user or agent messages
                # Provides domain-specific knowledge from microagents
                elif (
                    event.source == EventSource.USER
                    or event.source == EventSource.AGENT
                ) and event.recall_type == RecallType.KNOWLEDGE:
                    logger.debug(
                        f'Microagent knowledge recall from {event.source} message'
                    )
                    microagent_obs: RecallObservation | NullObservation | None = None

                    # Find and retrieve relevant microagent knowledge
                    microagent_obs = self._on_microagent_recall(event)
                    if microagent_obs is None:
                        microagent_obs = NullObservation(content='')

                    # Link this observation to the triggering action
                    # This releases the execution flow from waiting for the retrieval to complete
                    microagent_obs._cause = event.id  # type: ignore[union-attr]

                    # Add the observation to the event stream
                    self.event_stream.add_event(microagent_obs, EventSource.ENVIRONMENT)
                    return
        except Exception as e:
            # Handle any exceptions that occur during event processing
            error_str = f'Error: {str(e.__class__.__name__)}'
            logger.error(error_str)
            self.set_runtime_status(RuntimeStatus.ERROR_MEMORY, error_str)
            return

    def _on_workspace_context_recall(
        self, event: RecallAction
    ) -> RecallObservation | None:
        """
        Add repository and runtime information to the stream as a RecallObservation.

        This method collects comprehensive workspace context information including:
        1. Repository information (name, directory)
        2. Runtime information (available hosts, date, custom secrets)
        3. Repository instructions from all repo microagents
        4. Triggered microagent knowledge
        5. Conversation-specific instructions

        Multiple repo microagents are supported, and their contents will be concatenated
        with newlines between them to provide a comprehensive context.

        Args:
            event: The RecallAction event that triggered this recall

        Returns:
            RecallObservation with workspace context or None if no context is available
        """
        # Collect raw repository instructions from all repo microagents
        repo_instructions = ''
        for microagent in self.repo_microagents.values():
            if repo_instructions:
                repo_instructions += '\n\n'
            repo_instructions += microagent.content

        # Find any matched knowledge microagents based on the query
        microagent_knowledge = self._find_microagent_knowledge(event.query)

        # Create observation only if we have any contextual information to provide
        if (
            self.repository_info
            or self.runtime_info
            or repo_instructions
            or microagent_knowledge
            or self.conversation_instructions
        ):
            # Construct a comprehensive RecallObservation with all available context
            obs = RecallObservation(
                recall_type=RecallType.WORKSPACE_CONTEXT,
                # Repository information
                repo_name=self.repository_info.repo_name
                if self.repository_info and self.repository_info.repo_name is not None
                else '',
                repo_directory=self.repository_info.repo_directory
                if self.repository_info
                and self.repository_info.repo_directory is not None
                else '',
                # Repository instructions from microagents
                repo_instructions=repo_instructions if repo_instructions else '',
                # Runtime information
                runtime_hosts=self.runtime_info.available_hosts
                if self.runtime_info and self.runtime_info.available_hosts is not None
                else {},
                additional_agent_instructions=self.runtime_info.additional_agent_instructions
                if self.runtime_info
                and self.runtime_info.additional_agent_instructions is not None
                else '',
                date=self.runtime_info.date if self.runtime_info is not None else '',
                custom_secrets_descriptions=self.runtime_info.custom_secrets_descriptions
                if self.runtime_info is not None
                else {},
                # Microagent knowledge
                microagent_knowledge=microagent_knowledge,
                # Conversation instructions
                conversation_instructions=self.conversation_instructions.content
                if self.conversation_instructions is not None
                else '',
                # Content description
                content='Added workspace context',
            )
            return obs
        return None

    def _on_microagent_recall(
        self,
        event: RecallAction,
    ) -> RecallObservation | None:
        """
        Handle a knowledge recall request by finding and returning relevant microagent knowledge.

        This method is triggered when specific keywords in user or agent messages match
        the triggers defined in knowledge microagents. It creates a structured observation
        containing the relevant knowledge.

        Args:
            event: The RecallAction event that triggered this recall

        Returns:
            RecallObservation with microagent knowledge or None if no matching knowledge is found
        """
        # Find any matched microagents based on the query
        microagent_knowledge = self._find_microagent_knowledge(event.query)

        # Create observation only if we found matching microagent knowledge
        if microagent_knowledge:
            obs = RecallObservation(
                recall_type=RecallType.KNOWLEDGE,
                microagent_knowledge=microagent_knowledge,
                content='Retrieved knowledge from microagents',
            )
            return obs
        return None

    def _find_microagent_knowledge(self, query: str) -> list[MicroagentKnowledge]:
        """
        Find microagent knowledge based on a query string.

        This method searches all registered knowledge microagents for triggers that match
        the provided query. When a match is found, the microagent's content is included
        in the returned knowledge list.

        Args:
            query: The query string to search for microagent triggers

        Returns:
            A list of MicroagentKnowledge objects containing the name, trigger, and content
            of each matched microagent
        """
        recalled_content: list[MicroagentKnowledge] = []

        # Skip processing for empty queries
        if not query:
            return recalled_content

        # Search for microagent triggers in the query
        for name, microagent in self.knowledge_microagents.items():
            # Check if any trigger in this microagent matches the query
            trigger = microagent.match_trigger(query)
            if trigger:
                # Log the triggered microagent for debugging
                logger.info("Microagent '%s' triggered by keyword '%s'", name, trigger)

                # Add the microagent's knowledge to the result list
                recalled_content.append(
                    MicroagentKnowledge(
                        name=microagent.name,
                        trigger=trigger,
                        content=microagent.content,
                    )
                )
        return recalled_content

    def load_user_workspace_microagents(
        self, user_microagents: list[BaseMicroagent]
    ) -> None:
        """
        Load microagents from a user's cloned repository or workspace directory.

        This method is typically called from agent_session or setup once the workspace
        is cloned. It categorizes and stores the provided microagents based on their type
        (KnowledgeMicroagent or RepoMicroagent).

        Args:
            user_microagents: List of microagents found in the user's workspace
        """
        logger.info(
            'Loading user workspace microagents: %s', [m.name for m in user_microagents]
        )
        # Process each microagent and store it in the appropriate dictionary
        for user_microagent in user_microagents:
            if isinstance(user_microagent, KnowledgeMicroagent):
                # Store knowledge microagents (triggered by keywords)
                self.knowledge_microagents[user_microagent.name] = user_microagent
            elif isinstance(user_microagent, RepoMicroagent):
                # Store repository microagents (always active)
                self.repo_microagents[user_microagent.name] = user_microagent

    def _load_global_microagents(self) -> None:
        """
        Load microagents from the global microagents directory.

        Global microagents are installed with OpenHands and available to all users.
        They provide general-purpose knowledge and functionality.
        """
        # Load both repository and knowledge microagents from the global directory
        repo_agents, knowledge_agents = load_microagents_from_dir(
            GLOBAL_MICROAGENTS_DIR
        )

        # Store the loaded microagents in their respective dictionaries
        for name, agent_knowledge in knowledge_agents.items():
            self.knowledge_microagents[name] = agent_knowledge
        for name, agent_repo in repo_agents.items():
            self.repo_microagents[name] = agent_repo

    def _load_user_microagents(self) -> None:
        """
        Load microagents from the user's home directory.

        User microagents are stored in ~/.openhands/microagents/ and provide
        user-specific knowledge and functionality. This method creates the
        directory if it doesn't exist.
        """
        try:
            # Create the user microagents directory if it doesn't exist
            os.makedirs(USER_MICROAGENTS_DIR, exist_ok=True)

            # Load microagents from the user directory
            repo_agents, knowledge_agents = load_microagents_from_dir(
                USER_MICROAGENTS_DIR
            )

            # Store the loaded microagents in their respective dictionaries
            for name, agent_knowledge in knowledge_agents.items():
                self.knowledge_microagents[name] = agent_knowledge
            for name, agent_repo in repo_agents.items():
                self.repo_microagents[name] = agent_repo
        except Exception as e:
            logger.warning(
                f'Failed to load user microagents from {USER_MICROAGENTS_DIR}: {str(e)}'
            )

    def get_microagent_mcp_tools(self) -> list[MCPConfig]:
        """
        Get MCP tools configurations from all repository microagents.

        Repository microagents can define MCP (Model-Controller-Prompt) tools that
        extend the agent's capabilities. This method collects all such tool
        configurations from active repository microagents.

        Returns:
            A list of MCP tools configurations from microagents
        """
        mcp_configs: list[MCPConfig] = []

        # Check all repo microagents for MCP tools (always active)
        for agent in self.repo_microagents.values():
            if agent.metadata.mcp_tools:
                mcp_configs.append(agent.metadata.mcp_tools)
                logger.debug(
                    f'Found MCP tools in repo microagent {agent.name}: {agent.metadata.mcp_tools}'
                )

        return mcp_configs

    def set_repository_info(self, repo_name: str, repo_directory: str) -> None:
        """
        Store repository information for later inclusion in workspace context.

        Args:
            repo_name: The name of the repository
            repo_directory: The directory path of the repository
        """
        if repo_name or repo_directory:
            self.repository_info = RepositoryInfo(repo_name, repo_directory)
        else:
            self.repository_info = None

    def set_runtime_info(
        self,
        runtime: Runtime,
        custom_secrets_descriptions: dict[str, str],
    ) -> None:
        """
        Store runtime information for later inclusion in workspace context.

        This includes information about web hosts, ports, additional instructions,
        the current date, and descriptions of custom secrets.

        Args:
            runtime: The runtime environment containing web hosts and instructions
            custom_secrets_descriptions: Descriptions of custom secrets available to the agent
        """
        # Get the current UTC date
        utc_now = datetime.now(timezone.utc)
        date = str(utc_now.date())

        # Create RuntimeInfo with available information
        if runtime.web_hosts or runtime.additional_agent_instructions:
            self.runtime_info = RuntimeInfo(
                available_hosts=runtime.web_hosts,
                additional_agent_instructions=runtime.additional_agent_instructions,
                date=date,
                custom_secrets_descriptions=custom_secrets_descriptions,
            )
        else:
            self.runtime_info = RuntimeInfo(
                date=date,
                custom_secrets_descriptions=custom_secrets_descriptions,
            )

    def set_conversation_instructions(
        self, conversation_instructions: str | None
    ) -> None:
        """
        Set conversation-specific instructions for the agent.

        These instructions provide additional context or guidance for the agent
        during the conversation.

        Args:
            conversation_instructions: Instructions specific to the current conversation
        """
        self.conversation_instructions = ConversationInstructions(
            content=conversation_instructions or ''
        )

    def set_runtime_status(self, status: RuntimeStatus, message: str):
        """
        Send a runtime status update to the client.

        This method is used to report errors or status changes to the client
        through the status callback.

        Args:
            status: The runtime status to report
            message: A descriptive message about the status
        """
        if self.status_callback:
            try:
                # Get or create the event loop
                if self.loop is None:
                    self.loop = asyncio.get_running_loop()
                # Run the status update asynchronously
                asyncio.run_coroutine_threadsafe(
                    self._set_runtime_status('error', status, message), self.loop
                )
            except (RuntimeError, KeyError) as e:
                logger.error(
                    f'Error sending status message: {e.__class__.__name__}',
                    stack_info=False,
                )

    async def _set_runtime_status(
        self, msg_type: str, runtime_status: RuntimeStatus, message: str
    ):
        """
        Send a status message to the client asynchronously.

        Args:
            msg_type: The type of message (e.g., 'error')
            runtime_status: The runtime status to report
            message: A descriptive message about the status
        """
        if self.status_callback:
            self.status_callback(msg_type, runtime_status, message)
