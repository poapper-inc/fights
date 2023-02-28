"""
Puoribor, a variant of the classical `Quoridor <https://en.wikipedia.org/wiki/Quoridor>`_ game.
Coordinates are specified in the form of ``(x, y)``, where ``(0, 0)`` is the top left corner.
All coordinates and directions are absolute and does not change between agents.
Directions
    - Top: `+y`
    - Right: `+x`
    - Bottom: `-y`
    - Left: `-x`
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

if sys.version_info < (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias

from fights.base import BaseEnv, BaseState
from fights.envs.puoribor_cython import fast_step, legal_actions

PuoriborAction: TypeAlias = ArrayLike
"""
Alias of :obj:`ArrayLike` to describe the action type.
Encoded as an array of shape ``(3,)``, in the form of
[ `action_type`, `coordinate_x`, `coordinate_y` ].
`action_type`
    - 0 (move piece)
    - 1 (place wall horizontally)
    - 2 (place wall vertically)
    - 3 (rotate section)
`coordinate_x`, `coordinate_y`
    - position to move the piece to
    - top or left position to place the wall
    - top left position of the section to rotate
"""


@dataclass
class PuoriborState(BaseState):
    """
    ``PuoriborState`` represents the game state.
    """

    board: NDArray[np.int_]
    """
    Array of shape ``(C, W, H)``, where C is channel index and W, H is board width,
    height.
    Channels
        - ``C = 0``: one-hot encoded position of agent 0. (starts from top)
        - ``C = 1``: one-hot encoded position of agent 1. (starts from bottom)
        - ``C = 2``: label encoded positions of horizontal walls. (1 for wall placed
          by agent 0, 2 for agent 1)
        - ``C = 3``: label encoded positions of vertical walls. (encoding is same as
          ``C = 2``)
        - ``C = 4``: one-hot encoded positions of horizontal walls' midpoints.
        - ``C = 5``: one-hot encoded positions of vertical walls' midpoints.
    """

    walls_remaining: NDArray[np.int_]
    """
    Array of shape ``(2,)``, in the form of [ `agent0_remaining_walls`,
    `agent1_remaining_walls` ].
    """

    done: bool = False
    """
    Boolean value indicating whether the game is done.
    """

    def __str__(self) -> str:
        """
        Generate a human-readable string representation of the board.
        Uses unicode box drawing characters.
        """

        table_top = "┌───┬───┬───┬───┬───┬───┬───┬───┬───┐"
        vertical_wall = "│"
        vertical_wall_bold = "┃"
        horizontal_wall = "───"
        horizontal_wall_bold = "━━━"
        left_intersection = "├"
        middle_intersection = "┼"
        middle_intersection_bold = "╋"
        right_intersection = "┤"
        left_intersection_bottom = "└"
        middle_intersection_bottom = "┴"
        right_intersection_bottom = "┘"
        result = table_top + "\n"

        for y in range(9):
            board_line = self.board[:, :, y]
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
                    result += vertical_wall_bold
                elif x == 8:
                    result += vertical_wall
                else:
                    result += " "
                if x == 8:
                    result += "\n"
            result += left_intersection_bottom if y == 8 else left_intersection
            for x in range(9):
                board_cell = board_line[:, x]
                if board_cell[2]:
                    result += horizontal_wall_bold
                elif y == 8:
                    result += horizontal_wall
                else:
                    result += "   "
                if x == 8:
                    result += (
                        right_intersection_bottom if y == 8 else right_intersection
                    )
                else:
                    if np.any(self.board[4:, x, y]):
                        result += middle_intersection_bold
                    else:
                        result += (
                            middle_intersection_bottom
                            if y == 8
                            else middle_intersection
                        )
            result += "\n"

        return result

    def perspective(self, agent_id: int) -> NDArray[np.int_]:
        """
        Return board where specified agent with ``agent_id`` is on top.
        :arg agent_id:
            The ID of agent to use as base.
        :returns:
            A rotated ``board`` array. The board's channel 0 will contain position of
            agent of id ``agent_id``, and channel 1 will contain the opponent's
            position. In channel 2 and 3, walles labeled with 1 are set by agent of id
            ``agent_id``, and the others are set by the opponent.
        """
        if agent_id == 0:
            return self.board
        inverted_walls = (self.board[2:4] == 2).astype(np.int_) + (
            self.board[2:4] == 1
        ).astype(np.int_) * 2
        rotated = np.stack(
            [
                np.rot90(self.board[1], 2),
                np.rot90(self.board[0], 2),
                np.pad(
                    np.rot90(inverted_walls[0], 2)[:, 1:],
                    ((0, 0), (0, 1)),
                    constant_values=0,
                ),
                np.pad(
                    np.rot90(inverted_walls[1], 2)[1:],
                    ((0, 1), (0, 0)),  # type: ignore
                    constant_values=0,
                ),
                np.pad(
                    np.rot90(self.board[4], 2)[1:, 1:],
                    ((0, 1), (0, 1)),  # type: ignore
                    constant_values=0,
                ),
                np.pad(
                    np.rot90(self.board[5], 2)[1:, 1:],
                    ((0, 1), (0, 1)),  # type: ignore
                    constant_values=0,
                ),
            ]
        )
        return rotated

    def to_dict(self) -> Dict:
        """
        Serialize state object to dict.
        :returns:
            A serialized dict.
        """
        return {
            "board": self.board.tolist(),
            "walls_remaining": self.walls_remaining.tolist(),
            "done": self.done,
        }

    @staticmethod
    def from_dict(serialized) -> PuoriborState:
        """
        Deserialize from serialized dict.
        :arg serialized:
            A serialized dict.
        :returns:
            Deserialized ``PuoriborState`` object.
        """
        return PuoriborState(
            board=np.array(serialized["board"]),
            walls_remaining=np.array(serialized["walls_remaining"]),
            done=serialized["done"],
        )


class PuoriborEnv(BaseEnv[PuoriborState, PuoriborAction]):
    env_id = ("puoribor", 3)  # type: ignore
    """
    Environment identifier in the form of ``(name, version)``.
    """

    board_size: int = 9
    """
    Size (width and height) of the board.
    """

    max_walls: int = 10
    """
    Maximum allowed walls per agent.
    """

    def step(
        self,
        state: PuoriborState,
        agent_id: int,
        action: PuoriborAction,
        *,
        pre_step_fn: Optional[
            Callable[[PuoriborState, int, PuoriborAction], None]
        ] = None,
        post_step_fn: Optional[
            Callable[[PuoriborState, int, PuoriborAction], None]
        ] = None,
    ) -> PuoriborState:
        """
        Step through the game, calculating the next state given the current state and
        action to take.
        :arg state:
            Current state of the environment.
        :arg agent_id:
            ID of the agent that takes the action. (``0`` or ``1``)
        :arg action:
            Agent action, encoded in the form described by :obj:`PuoriborAction`.
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

        next_information = fast_step(
            state.board,
            state.walls_remaining,
            agent_id,
            np.array(action, dtype=np.int_),
            self.board_size,
        )

        next_state = PuoriborState(
            board=next_information[0],
            walls_remaining=next_information[1],
            done=bool(next_information[2]),
        )

        if post_step_fn is not None:
            post_step_fn(next_state, agent_id, action)
        return next_state

    def legal_actions(self, state: PuoriborState, agent_id: int) -> NDArray[np.int_]:
        """
        Find possible actions for the agent.

        :arg state:
            Current state of the environment.
        :arg agent_id:
            Agent_id of the agent.

        :returns:
            A numpy array of shape (4, 9, 9) which is one-hot encoding of possible actions.
        """
        return legal_actions(state, agent_id, self.board_size)

    def _check_in_range(
        self, pos: tuple, bottom_right: Optional[int] = None
    ) -> np.bool_:
        if bottom_right is None:
            bottom_right = self.board_size
        return (0 <= pos[0] < bottom_right) and (0 <= pos[1] < bottom_right)

    def _check_wall_blocked(
        self,
        board: NDArray[np.int_],
        current_pos: tuple,
        new_pos: tuple,
    ) -> np.bool_:
        if new_pos[0] > current_pos[0]:
            return np.any(board[3, current_pos[0] : new_pos[0], current_pos[1]])
        if new_pos[0] < current_pos[0]:
            return np.any(board[3, new_pos[0] : current_pos[0], current_pos[1]])
        if new_pos[1] > current_pos[1]:
            return np.any(board[2, current_pos[0], current_pos[1] : new_pos[1]])
        if new_pos[1] < current_pos[1]:
            return np.any(board[2, current_pos[0], new_pos[1] : current_pos[1]])
        return np.False_

    def _check_wins(self, board: NDArray[np.int_]) -> np.bool_:
        return board[0, :, -1].any() or board[1, :, 0].any()

    def initialize_state(self) -> PuoriborState:
        """
        Initialize a :obj:`PuoriborState` object with correct environment parameters.
        :returns:
            Created initial state object.
        """
        if self.board_size % 2 == 0:
            raise ValueError(
                f"cannot center pieces with even board_size={self.board_size}, please "
                "initialize state manually"
            )

        starting_pos_0 = np.zeros((self.board_size, self.board_size), dtype=np.int_)
        starting_pos_0[(self.board_size - 1) // 2, 0] = 1

        starting_board = np.stack(
            [
                np.copy(starting_pos_0),
                np.fliplr(starting_pos_0),
                np.zeros((self.board_size, self.board_size), dtype=np.int_),
                np.zeros((self.board_size, self.board_size), dtype=np.int_),
                np.zeros((self.board_size, self.board_size), dtype=np.int_),
                np.zeros((self.board_size, self.board_size), dtype=np.int_),
            ]
        )

        new_state = PuoriborState(
            board=starting_board,
            walls_remaining=np.array((self.max_walls, self.max_walls)),
            done=False,
        )

        return new_state
