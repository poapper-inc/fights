"""
Quoridor ai example based on minimax tree search + alpha_beta prunning.
"""

from collections import deque
from math import sqrt

import numpy as np

from fights.base import BaseAgent
from fights.envs import quoridor


class AlphabetaAgent(BaseAgent):
    env_id = ("quoridor", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)
        self.depth = 2

    def _all_actions(self):
        actions = []
        for coordinate_x in range(quoridor.QuoridorEnv.board_size):
            for coordinate_y in range(quoridor.QuoridorEnv.board_size):
                actions.append([0, coordinate_x, coordinate_y])
        for coordinate_x in range(quoridor.QuoridorEnv.board_size - 1):
            for coordinate_y in range(quoridor.QuoridorEnv.board_size - 1):
                actions.append([1, coordinate_x, coordinate_y])
                actions.append([2, coordinate_x, coordinate_y])
        return actions

    def _agent_pos(self, state: quoridor.QuoridorState, agent_id: int):
        for c_x in range(quoridor.QuoridorEnv.board_size):
            for c_y in range(quoridor.QuoridorEnv.board_size):
                if state.board[agent_id][c_x][c_y] == 1:
                    return c_x, c_y

    def _agent_dis_to_end(self, state: quoridor.QuoridorState, agent_id: int):
        distance = quoridor.QuoridorEnv.board_size * quoridor.QuoridorEnv.board_size
        visited = [
            [0 for _ in range(quoridor.QuoridorEnv.board_size)]
            for _ in range(quoridor.QuoridorEnv.board_size)
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
            if now[0] < quoridor.QuoridorEnv.board_size - 1:
                if state.board[3][now[0]][now[1]] == 0:
                    if visited[now[0] + 1][now[1]] == 0:
                        visited[now[0] + 1][now[1]] = visited[now[0]][now[1]] + 1
                        queue.append((now[0] + 1, now[1]))
            if now[1] < quoridor.QuoridorEnv.board_size - 1:
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

    def _evaluation(self, state: quoridor.QuoridorState):
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

    def __call__(self, state: quoridor.QuoridorState) -> quoridor.QuoridorAction:
        actions = self._all_actions()

        def search(
            state: quoridor.QuoridorState,
            agent_id: int,
            depth: int,
            lower_bound: int,
            upper_bound: int,
        ):
            if state.done:
                if agent_id == self.agent_id:
                    return -100000000
                else:
                    return 100000000
            if depth <= 0:
                return self._evaluation(state)

            new_agent_id = agent_id + 1
            new_depth = depth - 1
            if new_agent_id == 2:
                new_agent_id = 0

            if agent_id == self.agent_id:
                best_score = -100000000
            else:
                best_score = 100000000
            for action in actions:
                try:
                    new_state = quoridor.QuoridorEnv().step(state, agent_id, action)
                except:
                    ...
                else:
                    if agent_id == self.agent_id:
                        score = search(
                            new_state, new_agent_id, new_depth, best_score, upper_bound
                        )
                        if best_score < score:
                            best_score = score
                            if best_score > upper_bound:
                                break
                    else:
                        score = search(
                            new_state, new_agent_id, new_depth, best_score, upper_bound
                        )
                        if best_score > score:
                            best_score = score
                            if best_score < lower_bound:
                                break
            return best_score

        max_score = -100000001
        best_actions = []
        for action in actions:
            try:
                new_state = quoridor.QuoridorEnv().step(state, self.agent_id, action)
            except:
                ...
            else:
                score = search(
                    new_state, 1 - self.agent_id, self.depth - 1, max_score, 100000001
                )
                if score == max_score:
                    best_actions.append(action)
                elif score > max_score:
                    best_actions = [action]
                    max_score = score

        return self._rng.choice(best_actions)

