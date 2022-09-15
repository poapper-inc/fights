"""
Puoribor game example.
Prints board state to stdout with random agents by default.

Run `python puoribor.py -h` for more information.
"""

import argparse
import re
import sys
import time
from typing import Optional

sys.path.append("../")

import colorama
import numpy as np
from colorama import Fore, Style
from msgpack import Timestamp, packb

from fights.base import BaseAgent
from fights.envs import puoribor


class PuoriborAgent(BaseAgent):
    env_id = ("puoribor", 2)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
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


class Logger:
    log = []

    def __call__(
        self,
        state: puoribor.PuoriborState,
        agent_id: Optional[int],
        action: Optional[puoribor.PuoriborAction],
    ) -> None:
        self.log.append(
            {
                "state": state.to_dict(),
                "action": action if action is None else action.tolist(),  # type: ignore
                "agent_id": agent_id,
                "timestamp": Timestamp(time.time_ns()),
            }
        )


def fallback_to_ascii(s: str) -> str:
    try:
        s.encode(sys.stdout.encoding)
    except UnicodeEncodeError:
        s = re.sub("[┌┬┐├┼┤└┴┘╋]", "+", re.sub("[─━]", "-", re.sub("[│┃]", "|", s)))
    return s


def colorize_walls(s: str) -> str:
    return s.replace("━", Fore.BLUE + "━" + Style.RESET_ALL).replace(
        "┃", Fore.RED + "┃" + Style.RESET_ALL
    )


def run():
    assert puoribor.PuoriborEnv.env_id == PuoriborAgent.env_id
    colorama.init()

    state = puoribor.PuoriborEnv().initialize_state()
    agents = [PuoriborAgent(0), PuoriborAgent(1)]

    if not args.silent:
        print("\x1b[2J")

    it = 0
    logger = Logger()
    logger(state, None, None)
    while not state.done:
        if not args.silent:
            print("\x1b[1;1H")
            print(fallback_to_ascii(colorize_walls(str(state))))
        for agent in agents:
            action = agent(state)
            state = puoribor.PuoriborEnv().step(
                state, agent.agent_id, action, post_step_fn=logger
            )
            if not args.silent:
                print("\x1b[1;1H")
                print(fallback_to_ascii(colorize_walls(str(state))))
            if state.done:
                if not args.silent:
                    print(f"agent {agent.agent_id} won in {it} iters")
                break
        it += 1

    return logger.log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Puoribor example game")
    parser.add_argument(
        "-o",
        "--out",
        dest="out",
        help="output file path",
        required=False,
        type=argparse.FileType("wb"),
    )
    parser.add_argument(
        "-s",
        "--silent",
        dest="silent",
        action="store_true",
        help="silence board output",
        required=False,
        default=False,
    )
    args = parser.parse_args()

    history = run()

    if args.out is not None:
        with args.out as file:
            file.write(packb(history))
