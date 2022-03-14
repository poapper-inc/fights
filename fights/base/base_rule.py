from abc import ABC, abstractmethod


class BaseRule(ABC):
    def __init__(self, space):
        self.__space = space

    @abstractmethod
    def check_action_validity(self, action):
        ...

    def check_game_is_over(self, turn):
        ...
