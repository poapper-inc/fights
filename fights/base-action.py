from abc import ABC, abstractmethod
from fights import BaseState, BaseAction
from typing import List

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
    def getPossibleActions(state : BaseState) -> List[BaseAction]:
        #change to overriding is better?
        pass
    
    @abstractmethod
    def __hash__(self):
        pass
    
    @abstractmethod
    def __str__(self):
        pass    