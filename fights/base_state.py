from abc import ABC, abstractmethod


class BaseState(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    @staticmethod
    def default():
        pass

    @abstractmethod
    def __hash__(self):
        pass
