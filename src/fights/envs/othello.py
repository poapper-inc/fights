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

import sys
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

from . import othello_cythonfn

OthelloAction: TypeAlias = ArrayLike
"""
Alias of :obj:'ArrayLike' to describe the action type.
Encoded as an array of shape ''(2,)'',
in the form of [ 'coordinate_r', 'coordinate_c' ].
* Note that the action [3, 3] is jumping action, not putting a stone on board (3, 3).
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
    * Note that (W=3, H=3) is 1 when the agent can only jump

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

        return np.flip(np.rot90(self.board, 2, axes=(1, 2)), axis=0)

    def to_dict(self) -> dict:
        """
        Serialize state object to dict.
        :returns:
            A serialized dict.
        """
        return {
            "board": self.board.tolist(),
            "legal_actions": self.legal_actions.tolist(),
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

        action = np.array(action, dtype=np.int_)
        next_information = othello_cythonfn.fast_step(
            state.board,
            state.legal_actions,
            agent_id,
            action[0],
            action[1],
            self.board_size,
        )

        next_state = OthelloState(
            board=next_information[0],
            legal_actions=next_information[1],
            reward=np.array([next_information[2], next_information[3]]),
            done=bool(next_information[4]),
        )

        if post_step_fn is not None:
            post_step_fn(next_state, agent_id, action)

        return next_state

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

        legal_actions = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ]
        )

        initial_state = OthelloState(
            board=board,
            legal_actions=legal_actions,
            done=False,
            reward=np.zeros((2,), dtype=np.int_),
        )

        return initial_state
