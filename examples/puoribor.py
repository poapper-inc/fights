import sys

sys.path.append("../")

import os
import re

import numpy as np
from colorama import Fore, Style, init
from numpy.typing import NDArray

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


def fallback_to_ascii(s: str) -> str:
    try:
        s.encode(sys.stdout.encoding)
    except UnicodeEncodeError:
        s = re.sub("[┌┬┐├┼┤└┴┘]", "+", re.sub("[─━]", "-", re.sub("[│┃]", "|", s)))
    return s


def get_printable_board(board: NDArray[np.int_]) -> str:
    table_top = fallback_to_ascii("┌───┬───┬───┬───┬───┬───┬───┬───┬───┐")
    vertical_wall = fallback_to_ascii("│")
    vertical_wall_bold = fallback_to_ascii("┃")
    horizontal_wall = fallback_to_ascii("───")
    horizontal_wall_bold = fallback_to_ascii("━━━")
    left_intersection = fallback_to_ascii("├")
    middle_intersection = fallback_to_ascii("┼")
    right_intersection = fallback_to_ascii("┤")
    left_intersection_bottom = fallback_to_ascii("└")
    middle_intersection_bottom = fallback_to_ascii("┴")
    right_intersection_bottom = fallback_to_ascii("┘")
    result = table_top + "\n"

    for y in range(9):
        board_line = board[:, :, y]
        result += vertical_wall
        for x in range(9):
            board_cell = board_line[:, x]
            if board_cell[0]:
                result += " 0 "
            elif board_cell[1]:
                result += " 1 "
            else:
                result += "   "
            if board_cell[3]:
                result += Fore.RED + vertical_wall_bold + Style.RESET_ALL
            else:
                result += vertical_wall
            if x == 8:
                result += "\n"
        result += left_intersection_bottom if y == 8 else left_intersection
        for x in range(9):
            board_cell = board_line[:, x]
            if board_cell[2]:
                result += Fore.BLUE + horizontal_wall_bold + Style.RESET_ALL
            else:
                result += horizontal_wall
            if x == 8:
                result += right_intersection_bottom if y == 8 else right_intersection
            else:
                result += middle_intersection_bottom if y == 8 else middle_intersection
        result += "\n"

    return result


if __name__ == "__main__":
    assert puoribor.PuoriborEnv.env_id == PuoriborAgent.env_id

    init()
    state = puoribor.PuoriborEnv().initialize_state()
    agents = [PuoriborAgent(0), PuoriborAgent(1)]
    print(get_printable_board(state.board), end="")

    it = 0
    while not state.done:
        os.system("cls" if os.name == "nt" else "clear")
        print(get_printable_board(state.board), end="")
        for agent in agents:
            action = agent(state)
            state = puoribor.PuoriborEnv().step(state, agent.agent_id, action)
            os.system("cls" if os.name == "nt" else "clear")
            print(get_printable_board(state.board), end="")
            if state.done:
                print(f"agent {agent.agent_id} won in {it} iters")
                break
        it += 1
