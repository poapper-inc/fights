import warnings
from abc import ABCMeta, abstractmethod


class BaseEnv(metaclass=ABCMeta):
    agents = None
    possible_agents = None
    observation_spaces = None
    action_spaces = None
    rewards = None
    _cumulative_rewards = None
    dones = None
    infos = None

    def __init__(self):
        pass

    @abstractmethod
    def step(self, action):
        """
        Accepts and executes the action of the current agent_selection
        in the environment, automatically switches control to the next agent.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Resets the environment and sets it up for use when called the first time.
        """
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed=None):
        """
        Reseeds the environment (making the resulting environment deterministic).
        reset() must be called after seed(), and before step().
        """
        pass

    @abstractmethod
    def observe(self, agent):
        """
        Returns the observation an agent currently can make.
        last() calls this function.
        """
        raise NotImplementedError

    @abstractmethod
    def state(self):
        """
        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX.
        TODO: Require implementation
        """
        pass

    @abstractmethod
    def observation_space(self, agent):
        """
        A function that retrieves the observation space for a particular agent.
        This space should never change for a particular agent ID.

        Default implementation is to return the observation_spaces dict.
        """
        warnings.warn(
            "Your environment should override the observation_space function."
            "Attempting to use the observation_spaces dict attribute."
        )
        return self.observation_spaces[agent]

    @abstractmethod
    def action_space(self, agent):
        """
        A function that retrieves the action space for a particular agent.
        This space should never change for a particular agent ID.

        Default implementation is to return the action_spaces dict.
        """
        warnings.warn(
            "Your environment should override the action_space function."
            "Attempting to use the action_spaces dict attribute."
        )
        return self.action_spaces[agent]

    @property
    def num_agents(self):
        return len(self.agents)

    @property
    def max_num_agents(self):
        return len(self.possible_agents)

    def _dones_step_first(self):
        """
        Makes .agent_selection point to first done agent. Stores old value of agent_selection
        so that _was_done_step can restore the variable after the done agent steps.
        """
        _dones_order = [agent for agent in self.agents if self.dones[agent]]
        if _dones_order:
            self._skip_agent_selection = self.agent_selection
            self.agent_selection = _dones_order[0]
        return self.agent_selection

    def _clear_rewards(self):
        """
        Clears all items in .rewards
        """
        for agent in self.rewards:
            self.rewards[agent] = 0

    def _accumulate_rewards(self):
        """
        Adds .rewards dictionary to ._cumulative_rewards dictionary.
        Typically called near the end of a step() method.
        """
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

    @abstractmethod
    def agent_iter(self, max_iter=2**63):
        """
        Yields the current agent (self.agent_selection) when used in a loop where you step() each iteration.
        """
        return BaseIterable(self, max_iter)

    @abstractmethod
    def last(self, observe=True):
        """
        Returns observation, cumulative reward, done, info   for the current agent (specified by self.agent_selection)
        """
        agent = self.agent_selection
        observation = self.observe(agent) if observe else None
        return observation, self._cumulative_rewards[agent], self.dones[agent], self.infos[agent]

    def _was_done_step(self, action):
        """
        Helper function that performs step() for done agents.
        Does the following:
        1. Removes done agent from .agents, .dones, .rewards, ._cumulative_rewards, and .infos
        2. Loads next agent into .agent_selection: if another agent is done, loads that one, otherwise load next live agent
        3. Clear the rewards dict
        Highly recommended to use at the beginning of step as follows:
        def step(self, action):
            if self.dones[self.agent_selection]:
                self._was_done_step()
                return
            # main contents of step
        """
        if action is not None:
            raise ValueError("when an agent is done, the only valid action is None")

        # removes done agent
        agent = self.agent_selection
        assert self.dones[agent], "an agent that was not done as attempted to be removed"
        del self.dones[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)

        # finds next done agent or loads next live agent (Stored in _skip_agent_selection)
        _dones_order = [agent for agent in self.agents if self.dones[agent]]
        if _dones_order:
            if getattr(self, '_skip_agent_selection', None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _dones_order[0]
        else:
            if getattr(self, '_skip_agent_selection', None) is not None:
                self.agent_selection = self._skip_agent_selection
            self._skip_agent_selection = None
        self._clear_rewards()

    def __str__(self):
        """
        Returns a name which looks like: "space_invaders_v1"
        """
        if hasattr(self, 'metadata'):
            return self.metadata.get('name', self.__class__.__name__)
        else:
            return self.__class__.__name__

    @property
    def unwrapped(self):
        return self


class BaseIterable(metaclass=ABCMeta):
    def __init__(self, env, max_iter):
        self.env = env
        self.max_iter = max_iter

    def __iter__(self):
        return BaseIterator(self.env, self.max_iter)


class BaseIterator(metaclass=ABCMeta):
    def __init__(self, env, max_iter):
        self.env = env
        self.iters_til_term = max_iter

    def __next__(self):
        if not self.env.agents or self.iters_til_term <= 0:
            raise StopIteration
        self.iters_til_term -= 1
        return self.env.agent_selection


# TODO: BaseParallelEnv
