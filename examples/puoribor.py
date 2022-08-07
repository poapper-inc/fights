import sys

sys.path.append("../")

from fights.envs import puoribor
import numpy as np


class PuoriborAgent:
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
    state = puoribor.PuoriborEnv().initialize_state()
    agent0 = PuoriborAgent(0)
    agent1 = PuoriborAgent(1)

    while not state.done:
        action0 = agent0(state)
        state = puoribor.PuoriborEnv().step(state, agent0.agent_id, action0)

        if state.done:
            print("agent 0 won")
            break

        action1 = agent1(state)
        state = puoribor.PuoriborEnv().step(state, agent1.agent_id, action1)

        if state.done:
            print("agent 1 won")
            break
