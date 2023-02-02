"""
Othello(Reversi) Environment
Coordinates are specified in the form of ''(r, c)'', where ''(0, 0)'' is the top left corner.
All coordinates and directions are absolute and does not change between agents.

Directions
    - Top: '-r'
    - Right: '+c'
    - Bottom: '+r'
    - Left: '-c'
"""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

if sys.version_info < (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias

from fights.base import BaseEnv, BaseState

OthelloAction: TypeAlias = ArrayLike
"""
Alias of :obj:'ArrayLike' to describe the action type.
Encoded as an array of shape ''(2,)'',
in the form of [ 'coordinate_r', 'coordinate_c' ].
"""


@dataclass
class OthelloState(BaseState):
    """
    ''OthelloState'' represents the game state.
    """

    board: NDArray[np.int_]
    """
    Array of shape ``(C, W, H)``,
    where C is channel index
    and W, H is board width, height.

    Channels
        - ''C = 0'': one-hot encoded stones of agent 0. (black)
        - ''C = 1'': one-hot encoded stones of agent 1. (white)
    """

    legal_actions: NDArray[np.int_]
    """
    Array of shape ''(C, W, H)'',
    where C is channel index
    and W, H is board width, height.

    Channels
        - ''C = 0'': one-hot encoded possible positions of agent 0. (black)
        - ''C = 1'': one-hot encoded possible positions of agent 1. (white)
    """

    reward: NDArray[np.int_]
    """
    Array of shape ''(2,)'',
    where each value indicates the reward of each agent.

    values
        - Win : 1
        - Lose : -1
        - Draw or not done yet : 0
    """

    legal_set: list[defaultdict[tuple[int, int], set[int]]]
    """
    List of length 2,
    where each element is an defaultdict which contains 'maybe' possible locations for
    each agent.

    defaultdict
        - key(tuple) : empty locations next to opponent's stone.
        - value(set) : directions where opponent's stone is next to the location.
    """

    legal_dict: list[dict]
    """
    List of length 2,
    where each element is an dictionary which contains possible locations for each agent.

    dict
        - key(tuple) : possible locations.
        - value(0~7) : a direction where opponent's stones which can be flipped are.
    """

    done: bool = False
    """
    Boolean value indicating wheter the game is done.
    """

    def __str__(self) -> str:
        """
        Generate a human-readable string representation of the board.
        Uses unicode box drawing characters.
        """

        table_top = "┌───┬───┬───┬───┬───┬───┬───┬───┐"
        vertical_wall = "│"
        horizontal_wall = "───"
        left_intersection = "├"
        middle_intersection = "┼"
        right_intersection = "┤"
        left_intersection_bottom = "└"
        middle_intersection_bottom = "┴"
        right_intersection_bottom = "┘"

        result = table_top + "\n"

        for r in range(8):
            board_line = self.board[:, r, :]
            result += vertical_wall
            for c in range(8):
                board_cell = board_line[:, c]
                if board_cell[0]:
                    result += " □ "
                elif board_cell[1]:
                    result += " ■ "
                else:
                    result += "   "
                if c == 7:
                    result += vertical_wall
                    result += "\n"
                else:
                    result += " "
            result += left_intersection_bottom if r == 7 else left_intersection
            for c in range(8):
                if r == 7:
                    result += horizontal_wall
                    result += (
                        right_intersection_bottom
                        if c == 7
                        else middle_intersection_bottom
                    )
                else:
                    result += "   "
                    result += right_intersection if c == 7 else middle_intersection

            result += "\n"

        return result

    def perspective(self, agent_id: int) -> NDArray[np.int_]:
        """
        Return board observed by the agent whose ID is agent_id.

        :arg agent_id:
            The ID of agent to use as base.

        :returns:
            The ''board'' channel 0 will contain stones of ''agent_id'',
            and channel 1 will contain stones of opponent.
            The ''legal_actions'' channel 0 will contain legal actions of ''agent_id'',
            and channel 1 will contain legal actions of opponent.
            ''done'' has no difference.
            ''reward'' will be reversed.
            Considering that every game starts with 4 stones of fixed position,
            it returns flipped ''board'' array if ''agent_id'' is 1, and the same goes
            for ''legal_actions'' array.
        """

        if agent_id == 0:
            return self.board

        reverse_board = np.stack([np.fliplr(self.board[1]), np.fliplr(self.board[0])])
        reverse_state = reverse_board

        return reverse_state

    def need_jump(self, agent_id: int) -> bool:
        """
        Return whether the agent has no legal action.
        """
        return len(self.legal_dict[agent_id]) == 0

    def to_dict(self) -> dict:
        """
        Serialize state object to dict.
        :returns:
            A serialized dict.
        """
        return {
            "board": self.board.tolist(),
            "legal_actions": self.legal_actions.tolist(),
            "legal_set": [
                {" ".join([str(x) for x in k]): list(v) for k, v in d.items()}
                for d in self.legal_set
            ],
            "legal_dict": self.legal_dict,
            "done": self.done,
            "reward": self.reward.tolist(),
        }

    @staticmethod
    def from_dict(serialized) -> OthelloState:
        """
        Deserialize from serialized dict.
        :arg serialized:
            A serialized dict.
        :returns:
            Deserialized ``PuoriborState`` object.
        """
        return OthelloState(
            board=np.array(serialized["board"]),
            legal_actions=np.array(serialized["legal_actions"]),
            legal_set=[
                defaultdict(
                    set,
                    {
                        tuple(
                            [int(x) for x in k.split()],  # type: ignore
                        ): set(v)
                        for k, v in d.items()
                    },
                )
                for d in serialized["legal_set"]  # type: ignore
            ],
            legal_dict=serialized["legal_dict"],
            done=serialized["done"],
            reward=np.array(serialized["reward"]),
        )


class OthelloEnv(BaseEnv[OthelloState, OthelloAction]):
    env_id = ("othello", 0)  # type: ignore
    """
    Environment identifier in the form of ''(name, version)''.
    """

    board_size: int = 8
    """
    Size (width and height) of the board.
    """

    def step(
        self,
        state: OthelloState,
        agent_id: int,
        action: OthelloAction,
        *,
        pre_step_fn: Optional[
            Callable[[OthelloState, int, OthelloAction], None]
        ] = None,
        post_step_fn: Optional[
            Callable[[OthelloState, int, OthelloAction], None]
        ] = None,
    ) -> OthelloState:
        """
        Step through the game,
        calculating the next state given the current state and action to take.

        :arg state:
            Current state of the environment.

        :arg action:
            ID of the agent that takes the action. (''0'' or ''1'')

        :arg action:
            Agent action, encoded in the form described by :obj:'OthelloAction'.

        :arg pre_step_fn:
            Callback to run before executing action. ``state``, ``agent_id`` and
            ``action`` will be provided as arguments.

        :arg post_step_fn:
            Callback to run after executing action. The calculated state, ``agent_id``
            and ``action`` will be provided as arguments.

        :returns:
            A copy of the object with the restored state.
        """

        if pre_step_fn is not None:
            pre_step_fn(state, agent_id, action)

        action = np.asanyarray(action).astype(np.int_)
        r, c = action

        if not self._check_in_range(np.array([r, c])):
            raise ValueError(f"out of board: {(r, c)}")
        if not 0 <= agent_id <= 1:
            raise ValueError(f"invalid agent_id: {agent_id}")

        if state.legal_actions[agent_id][r][c] == 0:
            if state.board[1 - agent_id][r][c]:
                raise ValueError("cannot put a stone on opponent's stone")
            elif state.board[agent_id][r][c]:
                raise ValueError("cannot put a stone on another stone")
            else:
                raise ValueError("There is no stone to flip")

        new_board = np.copy(state.board)
        new_legal_set = copy.deepcopy(state.legal_set)
        new_legal_dict = copy.deepcopy(state.legal_dict)

        directions = (
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
        )

        # Put a stone and update its position of board, legal_dict, and legal_set.
        new_board[agent_id][r][c] = 1
        del new_legal_dict[agent_id][(r, c)]
        del new_legal_set[agent_id][(r, c)]
        if (r, c) in new_legal_set[1 - agent_id]:
            del new_legal_set[1 - agent_id][(r, c)]
        if (r, c) in new_legal_dict[1 - agent_id]:
            del new_legal_dict[1 - agent_id][(r, c)]

        # Update 8 surroundings(legal_set) of the location where the stone has just been put.
        for dir_id, dir in enumerate(directions):
            opp_dir_id = (dir_id + 4) % 8
            sur_r = r + dir[0]
            sur_c = c + dir[1]
            if not self._check_in_range(np.array([sur_r, sur_c])):
                continue
            if (
                new_board[agent_id][sur_r][sur_c] == 1
                or new_board[1 - agent_id][sur_r][sur_c] == 1
            ):
                continue
            new_legal_set[1 - agent_id][(sur_r, sur_c)].add(opp_dir_id)

        # Flip the stones and Update 8 surroundings(legal_set) of the locations where
        # stones flipped.
        # If one legal_set element is deleted, then verify same location of legal_dict
        # and delete it too if needed.
        for dir_id in state.legal_set[agent_id][(r, c)]:
            stones_to_flip = []
            temp_r = r
            temp_c = c
            for _ in range(1, self.board_size):
                temp_r += directions[dir_id][0]
                temp_c += directions[dir_id][1]
                if not self._check_in_range(np.array([temp_r, temp_c])):
                    break
                if state.board[1 - agent_id][temp_r][temp_c] == 1:
                    stones_to_flip.append((temp_r, temp_c))
                elif state.board[agent_id][temp_r][temp_c] == 1:
                    if stones_to_flip:
                        for (stone_r, stone_c) in stones_to_flip:
                            new_board[1 - agent_id][stone_r][stone_c] = 0
                            new_board[agent_id][stone_r][stone_c] = 1
                            for temp_dir_id, temp_dir in enumerate(directions):
                                opp_dir_id = (temp_dir_id + 4) % 8
                                if temp_dir_id == dir_id or opp_dir_id == dir_id:
                                    continue
                                sur_r = stone_r + temp_dir[0]
                                sur_c = stone_c + temp_dir[1]
                                if not self._check_in_range(np.array([sur_r, sur_c])):
                                    continue
                                if (
                                    new_board[agent_id][sur_r][sur_c] == 1
                                    or new_board[1 - agent_id][sur_r][sur_c] == 1
                                ):
                                    continue
                                new_legal_set[1 - agent_id][(sur_r, sur_c)].add(
                                    opp_dir_id
                                )
                                new_legal_set[agent_id][(sur_r, sur_c)].remove(
                                    opp_dir_id
                                )
                                if (sur_r, sur_c) in new_legal_dict[
                                    agent_id
                                ] and new_legal_dict[agent_id][
                                    (sur_r, sur_c)
                                ] == opp_dir_id:
                                    del new_legal_dict[agent_id][(sur_r, sur_c)]
                                if len(new_legal_set[agent_id][(sur_r, sur_c)]) == 0:
                                    del new_legal_set[agent_id][(sur_r, sur_c)]
                    break
                else:
                    break

        # Update legal_dict according to new board and legal_set.
        for agent_id in range(2):
            for r, c in new_legal_set[agent_id]:
                if (r, c) in new_legal_dict[agent_id]:
                    dir = directions[new_legal_dict[agent_id][(r, c)]]
                    if not self._can_flip(new_board, r, c, agent_id, dir[0], dir[1]):
                        del new_legal_dict[agent_id][(r, c)]
                if (r, c) not in new_legal_dict[agent_id]:
                    for dir_id in new_legal_set[agent_id][(r, c)]:
                        if self._can_flip(
                            new_board,
                            r,
                            c,
                            agent_id,
                            directions[dir_id][0],
                            directions[dir_id][1],
                        ):
                            new_legal_dict[agent_id][(r, c)] = dir_id
                            break

        # Build new legal_actions with legal_dict
        new_legal_actions = np.zeros(
            (2, self.board_size, self.board_size), dtype=np.int_
        )
        for agent_id in range(2):
            for r in range(self.board_size):
                for c in range(self.board_size):
                    new_legal_actions[agent_id][r][c] = (
                        1 if (r, c) in new_legal_dict[agent_id] else 0
                    )

        done = False
        reward = np.zeros((2,), dtype=np.int_)
        if len(new_legal_dict[0]) == 0 and len(new_legal_dict[1]) == 0:
            done = True
            reward = self._check_wins(new_board)

        next_state = OthelloState(
            board=new_board,
            legal_actions=new_legal_actions,
            done=done,
            reward=reward,
            legal_set=new_legal_set,
            legal_dict=new_legal_dict,
        )

        if post_step_fn is not None:
            post_step_fn(next_state, agent_id, action)

        return next_state

    def _can_flip(
        self,
        board: NDArray[np.int_],
        r: int,
        c: int,
        agent_id: int,
        dir_r: int,
        dir_c: int,
    ) -> bool:
        something_to_flip = False
        for _ in range(1, self.board_size):
            r += dir_r
            c += dir_c
            if not self._check_in_range(np.array([r, c])):
                return False
            if board[1 - agent_id][r][c] == 1:
                something_to_flip = True
            elif board[agent_id][r][c] == 1:
                if something_to_flip:
                    return True
                return False
        return False

    def _check_wins(self, board: NDArray[np.int_]) -> NDArray[np.int_]:
        agent0_cnt = np.count_nonzero(board[0])
        agent1_cnt = np.count_nonzero(board[1])

        if agent0_cnt > agent1_cnt:
            return np.array([1, -1])
        elif agent0_cnt < agent1_cnt:
            return np.array([-1, 1])
        else:
            return np.array([0, 0])

    def _check_in_range(self, pos: NDArray[np.int_], bottom_right=None) -> np.bool_:
        if bottom_right is None:
            bottom_right = np.array([self.board_size, self.board_size])
        return np.all(np.logical_and(np.array([0, 0]) <= pos, pos < bottom_right))

    def build_state(self, board: NDArray[np.int_]) -> OthelloState:
        """
        Build a state(including legal_set, legal_dict, legal_actions, done and reward)
        from the current board information.

        :arg state:
            Current state of the environment.

        :returns:
            A state which board is same as the input.
        """

        legal_set = [
            defaultdict(set),
            defaultdict(set),
        ]  # type: list[defaultdict[tuple[int, int], set[int]]]
        legal_dict = [dict(), dict()]  # type: list[dict[tuple[int, int], int]]

        directions = (
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
        )

        for r in range(self.board_size):
            for c in range(self.board_size):

                if board[0][r][c] == 0 and board[1][r][c] == 0:

                    for agent_id in range(2):

                        for dir_id, dir in enumerate(directions):

                            flipped = False
                            something_to_flip = False
                            temp_r = r
                            temp_c = c
                            for _ in range(1, self.board_size):

                                temp_r += dir[0]
                                temp_c += dir[1]
                                if not self._check_in_range(np.array([temp_r, temp_c])):
                                    break
                                if board[1 - agent_id][temp_r][temp_c] == 1:
                                    legal_set[agent_id][(r, c)].add(dir_id)
                                    something_to_flip = True
                                elif board[agent_id][temp_r][temp_c] == 1:
                                    if something_to_flip:
                                        flipped = True
                                    break
                                else:
                                    break
                            if flipped:
                                legal_dict[agent_id][(r, c)] = dir_id

        legal_actions = np.zeros((2, self.board_size, self.board_size), dtype=np.int_)
        for agent_id in range(2):
            for r in range(self.board_size):
                for c in range(self.board_size):
                    legal_actions[agent_id][r][c] = (
                        1 if (r, c) in legal_dict[agent_id] else 0
                    )

        done = False
        reward = np.zeros((2,), dtype=np.int_)
        if len(legal_dict[0]) == 0 and len(legal_dict[1]) == 0:
            done = True
            reward = self._check_wins(board)

        new_state = OthelloState(
            board=board,
            legal_actions=legal_actions,
            done=done,
            reward=reward,
            legal_set=legal_set,
            legal_dict=legal_dict,
        )

        return new_state

    def initialize_state(self) -> OthelloState:
        """
        Initialize a :obj:'OthelloState' object with correct environment parameters.

        :returns:
            Created initial state object.
        """
        if self.board_size % 2 == 1:
            raise ValueError(
                f"cannot center pieces with odd board_size={self.board_size}, please "
                "initialize state manually"
            )

        board = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ]
        )

        initial_state = self.build_state(board)

        return initial_state
