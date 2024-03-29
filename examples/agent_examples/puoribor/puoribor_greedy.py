"""
Puoribor ai example based on greedy search.
"""

import numpy as np
from collections import deque
from math import sqrt

from fights.base import BaseAgent
from fights.envs import puoribor


class GreedyAgent(BaseAgent):
    env_id = ("puoribor", 3)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _all_actions(self):
        actions = []
        for coordinate_x in range(puoribor.PuoriborEnv.board_size):
            for coordinate_y in range(puoribor.PuoriborEnv.board_size):
                actions.append([0, coordinate_x, coordinate_y])
        for coordinate_x in range(puoribor.PuoriborEnv.board_size - 1):
            for coordinate_y in range(puoribor.PuoriborEnv.board_size - 1):
                actions.append([1, coordinate_x, coordinate_y])
                actions.append([2, coordinate_x, coordinate_y])
        for coordinate_x in range(puoribor.PuoriborEnv.board_size - 3):
            for coordinate_y in range(puoribor.PuoriborEnv.board_size - 3):
                actions.append([3, coordinate_x, coordinate_y])
        return actions

    def _agent_pos(self, state: puoribor.PuoriborState, agent_id: int):
        for c_x in range(puoribor.PuoriborEnv.board_size):
            for c_y in range(puoribor.PuoriborEnv.board_size):
                if state.board[agent_id][c_x][c_y] == 1:
                    return c_x, c_y

    def _agent_dis_to_end(self, state: puoribor.PuoriborState, agent_id: int):
        distance = puoribor.PuoriborEnv.board_size * puoribor.PuoriborEnv.board_size
        visited = [
            [0 for _ in range(puoribor.PuoriborEnv.board_size)]
            for _ in range(puoribor.PuoriborEnv.board_size)
        ]
        queue = deque()

        queue.append(self._agent_pos(state, agent_id))
        visited[queue[0][0]][queue[0][1]] = 1

        while len(queue) > 0:
            now = queue.popleft()
            if agent_id == 0 and now[1] == 8:
                return visited[now[0]][now[1]] - 1
            if agent_id == 1 and now[1] == 0:
                return visited[now[0]][now[1]] - 1
            if now[1] > 0:
                if state.board[2][now[0]][now[1] - 1] == 0:
                    if visited[now[0]][now[1] - 1] == 0:
                        visited[now[0]][now[1] - 1] = visited[now[0]][now[1]] + 1
                        queue.append((now[0], now[1] - 1))
            if now[0] < puoribor.PuoriborEnv.board_size - 1:
                if state.board[3][now[0]][now[1]] == 0:
                    if visited[now[0] + 1][now[1]] == 0:
                        visited[now[0] + 1][now[1]] = visited[now[0]][now[1]] + 1
                        queue.append((now[0] + 1, now[1]))
            if now[1] < puoribor.PuoriborEnv.board_size - 1:
                if state.board[2][now[0]][now[1]] == 0:
                    if visited[now[0]][now[1] + 1] == 0:
                        visited[now[0]][now[1] + 1] = visited[now[0]][now[1]] + 1
                        queue.append((now[0], now[1] + 1))
            if now[0] > 0:
                if state.board[3][now[0] - 1][now[1]] == 0:
                    if visited[now[0] - 1][now[1]] == 0:
                        visited[now[0] - 1][now[1]] = visited[now[0]][now[1]] + 1
                        queue.append((now[0] - 1, now[1]))

        return distance

    def _evaluation(self, state: puoribor.PuoriborState):
        mine, opps = self._agent_dis_to_end(
            state, self.agent_id
        ), self._agent_dis_to_end(state, 1 - self.agent_id)
        return (
            int(sqrt(opps * 15000))
            - int(sqrt(mine * 10000))
            + (
                state.walls_remaining[self.agent_id]
                - state.walls_remaining[1 - self.agent_id]
            )
            * 10
        )

    def __call__(self, state: puoribor.PuoriborState) -> puoribor.PuoriborAction:
        actions = self._all_actions()

        def search(state: puoribor.PuoriborState, agent_id: int):
            if state.done:
                if agent_id == self.agent_id:
                    return -100000000
                else:
                    return 100000000
            return self._evaluation(state)

        max_score = -100000001
        best_actions = []
        for action in actions:
            try:
                new_state = puoribor.PuoriborEnv().step(state, self.agent_id, action)
            except:
                ...
            else:
                score = search(new_state, 1 - self.agent_id)
                if score == max_score:
                    best_actions.append(action)
                elif score > max_score:
                    best_actions = [action]
                    max_score = score

        return self._rng.choice(best_actions)
