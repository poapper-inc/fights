from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, Optional, Tuple, TypeVar

from numpy.typing import ArrayLike

S = TypeVar("S", bound="BaseState")
"""
A type variable for state type with bound ``BaseState``
"""

A = TypeVar("A", bound=ArrayLike)
"""
A type variable for action type with bound ``ArrayLike``
"""


class BaseState(ABC):
    @staticmethod
    @abstractmethod
    def from_dict(serialized) -> "BaseState":
        """
        Deserialize from dict.
        """
        ...

    @abstractmethod
    def to_dict(self) -> Dict:
        """
        Serialize to dict.
        """
        ...

    @property
    @abstractmethod
    def done(self) -> bool:
        """
        Whether the game is finished.
        """
        ...


class BaseEnv(ABC, Generic[S, A]):
    @property
    @abstractmethod
    def env_id(self) -> Tuple[str, int]:
        """
        Environment identifier in the form of ``(name, version)``.
        """
        ...

    @abstractmethod
    def step(
        self,
        state: S,
        agent_id: int,
        action: A,
        *,
        pre_step_fn: Optional[Callable[[S, int, A], None]] = None,
        post_step_fn: Optional[Callable[[S, int, A], None]] = None,
    ) -> S:
        """
        Step through the environment.
        """
        ...

    @abstractmethod
    def initialize_state(self) -> S:
        """
        Initialize state.
        """
        ...


class BaseAgent(ABC, Generic[S, A]):
    @property
    @abstractmethod
    def env_id(self) -> Tuple[str, int]:
        """
        Environment identifier in the form of ``(name, version)``.
        """
        ...

    @abstractmethod
    def __init__(self, agent_id: int) -> None:
        """
        Initialize an agent.
        """
        ...

    @abstractmethod
    def __call__(self, state: S) -> A:
        """
        Return the calculated agent action based on state input.
        """
        ...
