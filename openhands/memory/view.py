"""
Memory View Module for OpenHands

This module provides the View class, which represents a filtered and processed
view of the event history that can be sent to the LLM. Views handle the semantics
of condensation events, ensuring that forgotten events are properly excluded and
summaries are inserted at the appropriate positions.

Technical Stack:
- Python 3.12+
- Pydantic for data modeling
- Type hints with overloaded methods for better IDE support
"""

from __future__ import annotations

from typing import overload

from pydantic import BaseModel

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.agent import CondensationAction, CondensationRequestAction
from openhands.events.event import Event
from openhands.events.observation.agent import AgentCondensationObservation


class View(BaseModel):
    """
    A linearly ordered view of events ready for processing by the LLM.

    The View class represents a filtered and processed subset of the event history
    that is ready to be sent to the LLM. It handles the semantics of condensation
    events, ensuring that forgotten events are properly excluded and summaries are
    inserted at the appropriate positions.

    Views are typically produced by a condenser and provide list-like access to
    the filtered events.
    """

    # The filtered list of events to be included in the LLM context
    events: list[Event]
    # Flag indicating whether there's a pending condensation request
    unhandled_condensation_request: bool = False

    def __len__(self) -> int:
        """
        Get the number of events in the view.

        Returns:
            The number of events
        """
        return len(self.events)

    def __iter__(self):
        """
        Iterate over the events in the view.

        Returns:
            An iterator over the events
        """
        return iter(self.events)

    # To preserve list-like indexing, we support both slicing and position-based indexing.
    # The @overload decorators provide type hints for different input types.

    @overload
    def __getitem__(self, key: slice) -> list[Event]: ...

    @overload
    def __getitem__(self, key: int) -> Event: ...

    def __getitem__(self, key: int | slice) -> Event | list[Event]:
        """
        Access events by index or slice, similar to a list.

        Args:
            key: Either an integer index or a slice

        Returns:
            Either a single Event (for integer keys) or a list of Events (for slices)

        Raises:
            ValueError: If the key type is not supported
        """
        if isinstance(key, slice):
            # Handle slice access (e.g., view[1:5])
            start, stop, step = key.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(key, int):
            # Handle integer access (e.g., view[3])
            return self.events[key]
        else:
            # Handle unsupported key types
            raise ValueError(f'Invalid key type: {type(key)}')

    @staticmethod
    def from_events(events: list[Event]) -> View:
        """
        Create a filtered view from a list of events, respecting condensation semantics.

        This method processes the raw event list to:
        1. Remove events that have been marked as forgotten by condensation actions
        2. Insert summary observations at the appropriate positions
        3. Track whether there are unhandled condensation requests

        Args:
            events: The raw list of events to process

        Returns:
            A View containing the filtered and processed events
        """
        # Collect IDs of events that should be excluded from the view
        forgotten_event_ids: set[int] = set()
        for event in events:
            if isinstance(event, CondensationAction):
                # Add all forgotten event IDs from this condensation
                forgotten_event_ids.update(event.forgotten)
                # Also forget the condensation action itself
                forgotten_event_ids.add(event.id)
            if isinstance(event, CondensationRequestAction):
                # Forget condensation request actions
                forgotten_event_ids.add(event.id)

        # Filter out forgotten events
        kept_events = [event for event in events if event.id not in forgotten_event_ids]

        # Find the most recent summary and its insertion position
        summary: str | None = None
        summary_offset: int | None = None

        # The relevant summary is always in the last condensation action
        for event in reversed(events):
            if isinstance(event, CondensationAction):
                if event.summary is not None and event.summary_offset is not None:
                    summary = event.summary
                    summary_offset = event.summary_offset
                    break

        # Insert the summary at the specified offset if available
        if summary is not None and summary_offset is not None:
            logger.info(f'Inserting summary at offset {summary_offset}')
            kept_events.insert(
                summary_offset, AgentCondensationObservation(content=summary)
            )

        # Check for unhandled condensation requests
        # These are requests that appear after the most recent condensation action
        unhandled_condensation_request = False
        for event in reversed(events):
            if isinstance(event, CondensationAction):
                # Found a condensation action before any request, so no unhandled requests
                break
            if isinstance(event, CondensationRequestAction):
                # Found a request before any condensation action, so it's unhandled
                unhandled_condensation_request = True
                break

        # Create and return the view
        return View(
            events=kept_events,
            unhandled_condensation_request=unhandled_condensation_request,
        )
