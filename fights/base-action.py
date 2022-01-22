from abc import ABC, abstractmethod
from typing import List

from fights import BaseAction, BaseState


class BaseAction(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    @staticmethod
    def default():
        pass

    @abstractmethod
    @staticmethod
    def get_possible_actions(state: BaseState) -> List[BaseAction]:
        # change to overriding is better?
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass
