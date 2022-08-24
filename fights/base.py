from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, Optional, Tuple, TypeVar

from numpy.typing import ArrayLike

S = TypeVar("S", bound="BaseState")


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


class BaseEnv(ABC, Generic[S]):
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
        action: ArrayLike,
        *,
        pre_step_fn: Optional[Callable[[S, int, ArrayLike], None]] = None,
        post_step_fn: Optional[Callable[[S, int, ArrayLike], None]] = None,
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


class BaseAgent(ABC, Generic[S]):
    @property
    @abstractmethod
    def env_id(self) -> Tuple[str, int]:
        """
        Environment identifier in the form of ``(name, version)``.
        """
        ...

    @property
    @abstractmethod
    def agent_id(self) -> int:
        """
        Agent identifier.
        """
        ...

    @abstractmethod
    def __call__(self, state: S) -> ArrayLike:
        """
        Return the calculated agent action based on state input.
        """
        ...
