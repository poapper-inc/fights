"""
Manually played puoribor game.
"""

import argparse
import os
import re
import sys
import time
from typing import Optional

sys.path.append("../")

import colorama
from colorama import Fore, Style
from msgpack import Timestamp, packb

from fights.envs import puoribor


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
                "timestamp": Timestamp.from_unix_nano(time.time_ns()),
            }
        )


def add_left_ticks(s: str) -> str:
    lines = s.split("\n")
    lines[:-1:2] = ["   " + l for l in lines[:-1:2]]
    lines[1:-1:2] = [f" {i} " + l for i, l in enumerate(lines[1:-1:2])]
    top_ticks = "\n     " + "   ".join(map(str, range(9)))
    lines.insert(0, top_ticks)
    return "\n".join(lines)


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


if __name__ == "__main__":
    colorama.init()
    parser = argparse.ArgumentParser(description="Manually played puoribor game")
    parser.add_argument(
        "-c",
        "--clear-screen",
        dest="clear",
        action="store_true",
        help="clear screen every turn",
        required=False,
        default=False,
    )
    parser.add_argument(
        "-o",
        "--out",
        dest="out",
        help="output file path",
        required=False,
        type=argparse.FileType("wb"),
    )
    args = parser.parse_args()

    env = puoribor.PuoriborEnv()
    state = env.initialize_state()
    turn = 0
    logger = Logger()
    logger(state, None, None)
    while not state.done:
        if args.clear:
            os.system("cls" if os.name == "nt" else "clear")
        print(fallback_to_ascii(colorize_walls(add_left_ticks(str(state)))))
        print()
        print(
            f"Agent {turn}'s turn (remaining wall count: {state.walls_remaining[turn]})"
        )
        valid_input = False
        while not valid_input:
            try:
                action = list(
                    map(int, input("action_id, x, y separated by space: ").split())
                )
                state = env.step(state, turn, action, post_step_fn=logger)
            except ValueError as err:
                print(f"Invalid move: {err}")
            else:
                valid_input = True
        print()

        if state.done:
            print(f"Agent {turn} won!")
            break

        turn += 1
        turn %= 2

    if args.out is not None:
        with args.out as file:
            file.write(packb(logger.log))
