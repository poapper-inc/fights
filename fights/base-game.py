from abc import ABC, abstractmethod
from multiprocessing.spawn import is_forking
from pettingzoo import AECEnv
from gym import spaces
from typing import List
import numpy as np
import functools
from pettingzoo.utils import agent_selector
from fights import BaseAction, BaseState, BaseAgent

from fights.util import raiseNotDefined



class BaseGame(ABC, AECEnv):
    def __init__( self, agents : List[BaseAgent], display, rules, numIters,
                 agentSpaces : spaces.Space, observationSpaces: spaces.Space, 
                 metadata = {'render.modes': ['human'], "name": "base_game"}, 
                 startingIndex=0, catchExceptions=False ):
        self.agentCrashed = False
        self.possible_agents = agents
        self.metadata = metadata
        self.display = display
        self.rules = rules
        self.numIters = numIters
        self.startingIndex = startingIndex
        self.gameOver = False
        self.catchExceptions = catchExceptions
        self.totalAgentTimes = [0 for agent in agents]
        self.totalAgentTimeWarnings = [0 for agent in agents]
        self.agentTimeout = False
        self.actionSpaces = {agent: agentSpaces for agent in self.possible_agents}
        self.observationSpaces = {agent: observationSpaces for agent in self.possible_agents}
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    @abstractmethod
    def render(self, mode="human"):
        pass
    
    @abstractmethod
    def close(self):
        pass
    
    @abstractmethod
    def setStateByAction(self, action : BaseAction):
        pass
    
    @abstractmethod
    def setRewardByState(self):
        pass
    
    def setObservations(self):
        for i in self.agents:
            self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
        
    def observe(self, agent):
        return np.array(self.observations[agent])
    
    def reset(self):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: BaseState.default() for agent in self.agents}
        self.observations = {agent: BaseState.default() for agent in self.agents}
        self.numMoves = 0
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
    
    def step(self, action : BaseAction):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        agent = self.agent_selection

        self._cumulative_rewards[agent] = 0
        #self.state[self.agent_selection] = action;
        self.setStateByAction(action)
        
        if self._agent_selector.is_last():
            self.setRewardByState()
            self.num_moves += 1
            self.dones = {agent: self.num_moves >= self.numIters for agent in self.agents}

            self.setObservations()
        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = BaseState.default()
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()


    
