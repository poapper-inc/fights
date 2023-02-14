"""
Quoridor Environment Benchmark
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import time

from fights.base import BaseAgent
from fights.envs import quoridor

class RandomAgent(BaseAgent):
    env_id = ("quoridor", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def __call__(self, state: quoridor.QuoridorState) -> quoridor.QuoridorAction:
        legal_actions_np = quoridor.QuoridorEnv().legal_actions(state, self.agent_id)
        return self._rng.choice(np.argwhere(legal_actions_np == 1))

def run():
    assert quoridor.QuoridorEnv.env_id == RandomAgent.env_id
    start = time.time()

    for game in range(10):

        print(game)

        state = quoridor.QuoridorEnv().initialize_state()
        agents = [RandomAgent(0, game), RandomAgent(1, game)]

        while not state.done:

            for agent in agents:

                action = agent(state)
                state = quoridor.QuoridorEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

    end = time.time()
    print(f"{end - start} sec")

if __name__ == "__main__":
    run()
    
    