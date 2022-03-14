from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self):
        self.__name = None

    def get_name(self):
        return self.__name

    @abstractmethod
    def determine_and_return_action(self, space):
        ...
