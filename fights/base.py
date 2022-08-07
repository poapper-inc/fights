import sys
from abc import ABC, abstractmethod
from typing import Any, Tuple

from numpy.typing import ArrayLike

if sys.version_info < (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias


State: TypeAlias = Any


class BaseEnv(ABC):
    @property
    @abstractmethod
    def env_id(self) -> Tuple[str, int]:
        """
        Environment identifier in the form of ``(name, version)``.
        """
        ...

    @abstractmethod
    def step(self, state: State, agent_id: int, action: ArrayLike) -> State:
        """
        Step through the environment.
        """
        ...

    @abstractmethod
    def initialize_state(self) -> Any:
        """
        Initialize state.
        """
        ...


class BaseAgent(ABC):
    @property
    @abstractmethod
    def env_id(self) -> Tuple[str, int]:
        """
        Environment identifier in the form of ``(name, version)``.
        """
        ...

    @abstractmethod
    def __call__(self, state: State) -> ArrayLike:
        """
        Return the calculated agent action based on state input.
        """
        ...
