from typing import Tuple, Type

from ..base import BaseEnv, BaseState
from .puoribor import PuoriborEnv, PuoriborState


def resolve(name: str) -> Tuple[Type[BaseEnv], Type[BaseState]]:
    mappings = {"puoribor": (PuoriborEnv, PuoriborState)}
    if name not in mappings:
        raise ValueError(f"environment with name {name} not found")
    return mappings[name]
