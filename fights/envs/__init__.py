from typing import Tuple, Type, cast

from ..base import BaseEnv, BaseState
from .puoribor import PuoriborEnv, PuoriborState


def resolve(name: str) -> Tuple[Type[BaseEnv[BaseState]], Type[BaseState]]:
    """
    Resolve environment and state classes with environment name.

    :arg name:
        The name of the environment to resolve.

    :returns:
        A tuple of (env class, env state).
    """
    mappings = {"puoribor": (PuoriborEnv, PuoriborState)}
    if name not in mappings:
        raise ValueError(f"environment with name {name} not found")
    return cast(Tuple[Type[BaseEnv[BaseState]], Type[BaseState]], mappings[name])
