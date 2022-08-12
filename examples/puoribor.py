import sys

sys.path.append("../")

import colorama
import numpy as np

from fights.base import BaseAgent
from fights.envs import puoribor


class PuoriborAgent(BaseAgent):
    env_id = ("puoribor", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: puoribor.PuoriborState):
        actions = []
        for action_type in [0, 1, 2, 3]:
            for coordinate_x in range(puoribor.PuoriborEnv.board_size):
                for coordinate_y in range(puoribor.PuoriborEnv.board_size):
                    action = [action_type, coordinate_x, coordinate_y]
                    try:
                        puoribor.PuoriborEnv().step(state, self.agent_id, action)
                    except:
                        ...
                    else:
                        actions.append(action)
        return actions

    def __call__(self, state: puoribor.PuoriborState) -> puoribor.PuoriborAction:
        actions = self._get_all_actions(state)
        return self._rng.choice(actions)


if __name__ == "__main__":
    assert puoribor.PuoriborEnv.env_id == PuoriborAgent.env_id
    colorama.init()
    state = puoribor.PuoriborEnv().initialize_state()
    agents = [PuoriborAgent(0), PuoriborAgent(1)]

    print("\x1b[2J")

    it = 0
    while not state.done:
        print("\x1b[1;1H")
        print(state)
        for agent in agents:
            action = agent(state)
            state = puoribor.PuoriborEnv().step(state, agent.agent_id, action)

            print("\x1b[1;1H")
            print(state)
            if state.done:
                print(f"agent {agent.agent_id} won in {it} iters")
                break
        it += 1
