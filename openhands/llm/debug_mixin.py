"""
OpenHands LLM 调试混入模块

本模块提供了调试和日志功能，用于记录 LLM 交互过程。
支持多模态内容的格式化和调试信息的结构化输出。

主要功能：
- 记录发送到 LLM 的提示内容
- 记录从 LLM 接收的响应内容
- 格式化多模态内容（文本和图像）
- 提供结构化的调试信息输出
- 支持视觉内容的调试显示

技术栈:
- Python 标准库: 类型注解和字符串处理
- OpenHands 日志系统: 结构化日志记录
"""

from typing import Any

from openhands.core.logger import llm_prompt_logger, llm_response_logger
from openhands.core.logger import openhands_logger as logger

MESSAGE_SEPARATOR = '\n\n----------\n\n'  # 消息分隔符


class DebugMixin:
    """
    调试功能混入类。

    该类提供了调试功能，用于记录 LLM 交互。它记录发送到 LLM 的提示和从 LLM 接收的响应，
    并格式化多模态内容以便于调试。

    技术栈:
    - Python 3.12+
    - 日志系统用于记录交互
    - 混入模式用于代码复用
    """

    def log_prompt(self, messages: list[dict[str, Any]] | dict[str, Any]) -> None:
        if not messages:
            logger.debug('No completion messages!')
            return

        messages = messages if isinstance(messages, list) else [messages]
        debug_message = MESSAGE_SEPARATOR.join(
            self._format_message_content(msg)
            for msg in messages
            if msg['content'] is not None
        )

        if debug_message:
            llm_prompt_logger.debug(debug_message)
        else:
            logger.debug('No completion messages!')

    def log_response(self, message_back: str) -> None:
        if message_back:
            llm_response_logger.debug(message_back)

    def _format_message_content(self, message: dict[str, Any]) -> str:
        content = message['content']
        if isinstance(content, list):
            return '\n'.join(
                self._format_content_element(element) for element in content
            )
        return str(content)

    def _format_content_element(self, element: dict[str, Any] | Any) -> str:
        if isinstance(element, dict):
            if 'text' in element:
                return str(element['text'])
            if (
                self.vision_is_active()
                and 'image_url' in element
                and 'url' in element['image_url']
            ):
                return str(element['image_url']['url'])
        return str(element)

    # This method should be implemented in the class that uses DebugMixin
    def vision_is_active(self) -> bool:
        raise NotImplementedError
