from abc import ABC, abstractmethod


class BaseEnv(ABC):
    def __init__(self, env_rule, env_mode):
        self.turn = None
        self.agent = []
        self.init_space()
        self.set_rule(env_rule)
        self.set_mode(env_mode)

    @abstractmethod
    def init_space(self):
        ...

    @abstractmethod
    def next_turn(self):
        ...

    def step(self):
        self.before_action()
        self.agent[self.turn].determine_and_return_action()
        self.after_action()
        self.next_turn()

    def before_action(self):
        pass

    def after_action(self):
        pass

    @abstractmethod
    def set_rule(self, env_rule):
        ...

    @abstractmethod
    def set_mode(self, env_mode):
        ...

    @abstractmethod
    def run(self):
        ...
