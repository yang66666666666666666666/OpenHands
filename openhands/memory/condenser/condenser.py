"""
Memory Condenser Module for OpenHands

This module provides the core functionality for condensing conversation history
to manage token limits and maintain context in LLM interactions. Condensers are
responsible for summarizing or filtering events to reduce token usage while
preserving important context.

Technical Stack:
- Python 3.12+
- Pydantic for data modeling
- Abstract base classes for extensible condenser implementations
- Factory pattern for condenser creation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel

from openhands.controller.state.state import State
from openhands.core.config.condenser_config import CondenserConfig
from openhands.events.action.agent import CondensationAction
from openhands.memory.view import View

CONDENSER_METADATA_KEY = 'condenser_meta'
"""Key identifying where metadata is stored in a `State` object's `extra_data` field."""


def get_condensation_metadata(state: State) -> list[dict[str, Any]]:
    """Utility function to retrieve a list of metadata batches from a `State`.

    Args:
        state: The state to retrieve metadata from.

    Returns:
        list[dict[str, Any]]: A list of metadata batches, each representing a condensation.
    """
    if CONDENSER_METADATA_KEY in state.extra_data:
        return state.extra_data[CONDENSER_METADATA_KEY]
    return []


CONDENSER_REGISTRY: dict[type[CondenserConfig], type[Condenser]] = {}
"""Registry of condenser configurations to their corresponding condenser classes."""


class Condensation(BaseModel):
    """
    Represents the result of a condensation operation.

    When a condenser decides to condense history, it produces a Condensation object
    containing a CondensationAction that will be added to the event stream. This
    action contains information about which events were condensed and their summary.
    """

    action: CondensationAction
    """The condensation action to be added to the event stream"""


class Condenser(ABC):
    """
    Abstract base class for all condenser implementations.

    Condensers are responsible for managing conversation history size by reducing
    the number of events that need to be included in the LLM context. They take a list
    of Event objects and reduce them into a potentially smaller list, typically by
    summarizing or filtering events.

    When the context window or token limit is exceeded, condensers start condensing
    chunks of messages into summaries. Each summary is then injected into the context
    in place of the events it summarizes.

    Usage:
    Agents call the `condensed_history` method on the current State to get a condensed
    view of the history. If the condenser returns a Condensation instead of a View,
    the agent should return Condensation.action instead of producing its own action.
    """

    def __init__(self):
        """Initialize the condenser with empty metadata containers."""
        # Stores metadata for the current condensation batch
        self._metadata_batch: dict[str, Any] = {}
        # Stores LLM-specific metadata
        self._llm_metadata: dict[str, Any] = {}

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add information to the current metadata batch.

        Any key/value pairs added to the metadata batch will be recorded in the
        State at the end of the current condensation. This is useful for tracking
        diagnostic information about the condensation process.

        Args:
            key: The key to store the metadata under
            value: The metadata to store
        """
        self._metadata_batch[key] = value

    def write_metadata(self, state: State) -> None:
        """
        Write the current batch of metadata to the State.

        This method stores the current metadata batch in the State's extra_data
        field and resets the batch for the next condensation.

        Args:
            state: The state to write metadata to
        """
        # Initialize the metadata list if it doesn't exist
        if CONDENSER_METADATA_KEY not in state.extra_data:
            state.extra_data[CONDENSER_METADATA_KEY] = []

        # Add the current batch if it's not empty
        if self._metadata_batch:
            state.extra_data[CONDENSER_METADATA_KEY].append(self._metadata_batch)

        # Reset the batch for the next condensation
        self._metadata_batch = {}

    @contextmanager
    def metadata_batch(self, state: State):
        """
        Context manager to ensure batched metadata is always written to the State.

        This ensures that metadata is written even if an exception occurs during
        the condensation process.

        Args:
            state: The state to write metadata to
        """
        try:
            yield
        finally:
            self.write_metadata(state)

    @abstractmethod
    def condense(self, View) -> View | Condensation:
        """
        Condense a sequence of events into a potentially smaller list.

        This abstract method must be implemented by concrete condenser classes.
        Each implementation should define its own condensation strategy.

        Args:
            View: A view of the history containing all events that should be condensed

        Returns:
            View | Condensation: Either a condensed view of the events or a
                                Condensation object indicating the history has been condensed
        """

    def condensed_history(self, state: State) -> View | Condensation:
        """
        Condense the state's history using the condenser's strategy.

        This is the main entry point for using a condenser. It extracts LLM metadata
        from the state, sets up the metadata batch context, and calls the condense method.

        Args:
            state: The state containing the history to condense

        Returns:
            View | Condensation: Either a condensed view of the events or a
                                Condensation object indicating the history has been condensed
        """
        # Extract LLM metadata from the state
        self._llm_metadata = state.to_llm_metadata('condenser')

        # Ensure metadata is written to the state after condensation
        with self.metadata_batch(state):
            return self.condense(state.view)

    @classmethod
    def register_config(cls, configuration_type: type[CondenserConfig]) -> None:
        """
        Register a new condenser configuration type in the global registry.

        This class method implements a factory pattern for condenser creation.
        Condenser implementations register their configuration types, allowing
        the system to create the appropriate condenser instance from a configuration.

        Args:
            configuration_type: The type of configuration used to create instances of the condenser

        Raises:
            ValueError: If the configuration type is already registered
        """
        if configuration_type in CONDENSER_REGISTRY:
            raise ValueError(
                f'Condenser configuration {configuration_type} is already registered'
            )
        CONDENSER_REGISTRY[configuration_type] = cls

    @classmethod
    def from_config(cls, config: CondenserConfig) -> Condenser:
        """
        Create a condenser instance from a configuration object.

        This factory method looks up the appropriate condenser class in the registry
        based on the configuration type and delegates to that class's from_config method.

        Args:
            config: Configuration for the condenser

        Returns:
            Condenser: A condenser instance of the appropriate type

        Raises:
            ValueError: If the condenser type is not recognized
        """
        try:
            # Look up the condenser class in the registry
            condenser_class = CONDENSER_REGISTRY[type(config)]
            # Delegate to the class's from_config method
            return condenser_class.from_config(config)
        except KeyError:
            raise ValueError(f'Unknown condenser config: {config}')


class RollingCondenser(Condenser, ABC):
    """
    Base class for condensers that apply condensation to a rolling history.

    RollingCondenser implements a specific condensation strategy that:
    1. Checks if condensation is needed using should_condense()
    2. If needed, generates a Condensation using get_condensation()
    3. Otherwise, returns the original view unchanged

    The rolling history is generated by View.from_events, which analyzes all events
    in the history and produces a View object representing what will be sent to the LLM.

    Subclasses must implement should_condense() and get_condensation() to define
    when and how condensation should occur.
    """

    @abstractmethod
    def should_condense(self, view: View) -> bool:
        """
        Determine if a view should be condensed.

        This method defines the condition that triggers condensation, such as
        exceeding a token limit or message count threshold.

        Args:
            view: The view to check for condensation

        Returns:
            True if condensation should be performed, False otherwise
        """

    @abstractmethod
    def get_condensation(self, view: View) -> Condensation:
        """
        Generate a Condensation from a view.

        This method defines how to create a condensed representation of the view,
        typically by summarizing or filtering events.

        Args:
            view: The view to condense

        Returns:
            A Condensation object containing the condensation action
        """

    def condense(self, view: View) -> View | Condensation:
        """
        Condense a view if necessary, based on the should_condense condition.

        This implementation follows a simple decision pattern:
        1. If should_condense() returns True, call get_condensation()
        2. Otherwise, return the original view unchanged

        Args:
            view: The view to potentially condense

        Returns:
            Either the original view or a Condensation object
        """
        # Check if condensation is needed
        if self.should_condense(view):
            # Generate and return the condensation
            return self.get_condensation(view)
        else:
            # Return the original view unchanged
            return view
