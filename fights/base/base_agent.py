from abc import ABCMeta, abstractmethod


class BaseAgent(metaclass=ABCMeta):
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def get_agent_id(self):
        return self.agent_id

    @abstractmethod
    def policy(self, observation):
        pass
