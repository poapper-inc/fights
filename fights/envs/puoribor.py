"""
Puoribor, a variant of the classical `Quoridor <https://en.wikipedia.org/wiki/Quoridor>`_ game.

Coordinates are specified in the form of ``(x, y)``, where ``(0, 0)`` is the bottom left corner.
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
from typing import Callable, Deque, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

if sys.version_info < (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias

from fights.base import BaseEnv, BaseState

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
        - ``C = 2``: one-hot encoded positions of horizontal walls.
        - ``C = 3``: one-hot encoded positions of vertical walls.
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
                    result += (
                        middle_intersection_bottom if y == 8 else middle_intersection
                    )
            result += "\n"

        return result

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


class PuoriborEnv(BaseEnv):
    env_id = ("puoribor", 1)  # type: ignore
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
        action: ArrayLike,
        *,
        pre_step_fn: Optional[Callable[[PuoriborState, int, ArrayLike], None]] = None,
        post_step_fn: Optional[Callable[[PuoriborState, int, ArrayLike], None]] = None,
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

        action = np.asanyarray(action).astype(np.int_)
        action_type, x, y = action
        if not self._check_in_range(np.array([x, y])):
            raise ValueError(f"out of board: {(x, y)}")
        if not 0 <= agent_id <= 1:
            raise ValueError(f"invalid agent_id: {agent_id}")

        board = np.copy(state.board)
        walls_remaining = np.copy(state.walls_remaining)

        if action_type == 0:  # Move piece
            current_pos = np.argwhere(state.board[agent_id] == 1)[0]
            new_pos = np.array([x, y])
            opponent_pos = np.argwhere(state.board[1 - agent_id] == 1)[0]
            if np.all(new_pos == opponent_pos):
                raise ValueError("cannot move to opponent's position")

            delta = new_pos - current_pos
            taxicab_dist = np.abs(delta).sum()
            if taxicab_dist == 0:
                raise ValueError("cannot move zero blocks")
            elif taxicab_dist > 2:
                raise ValueError("cannot move more than two blocks")
            elif (
                taxicab_dist == 2
                and np.any(delta == 0)
                and not np.all(current_pos + delta // 2 == opponent_pos)
            ):
                raise ValueError("cannot jump over nothing")

            if np.all(delta):  # If moving diagonally
                if np.any(current_pos + delta * [0, 1] != opponent_pos) and np.any(
                    current_pos + delta * [1, 0] != opponent_pos
                ):
                    # Only diagonal jumps are permitted.
                    # Agents cannot simply move in diagonal direction.
                    raise ValueError("cannot move diagonally")
                elif self._check_wall_blocked(board, current_pos, opponent_pos):
                    raise ValueError("cannot jump over walls")

                original_jump_pos = current_pos + 2 * (opponent_pos - current_pos)
                if self._check_in_range(
                    original_jump_pos
                ) and not self._check_wall_blocked(
                    board, current_pos, original_jump_pos
                ):
                    raise ValueError(
                        "cannot diagonally jump if linear jump is possible"
                    )
                elif self._check_wall_blocked(board, opponent_pos, new_pos):
                    raise ValueError("cannot jump over walls")
            elif self._check_wall_blocked(board, current_pos, new_pos):
                raise ValueError("cannot jump over walls")

            board[agent_id][tuple(current_pos)] = 0
            board[agent_id][tuple(new_pos)] = 1

        elif action_type == 1:  # Place wall horizontally
            if walls_remaining[agent_id] == 0:
                raise ValueError(f"no walls left for agent {agent_id}")
            if y == self.board_size - 1:
                raise ValueError("cannot place wall on the edge")
            elif x == self.board_size - 1:
                raise ValueError("right section out of board")
            elif np.any(board[2, x : x + 2, y]):
                raise ValueError("wall already placed")
            elif np.all(board[3, x, y : y + 2]):
                raise ValueError("cannot create intersecting walls")
            board[2, x, y] = 1
            board[2, x + 1, y] = 1
            if not self._check_path_exists(board, 0) or not self._check_path_exists(
                board, 1
            ):
                raise ValueError("cannot place wall blocking all paths")
            walls_remaining[agent_id] -= 1

        elif action_type == 2:  # Place wall vertically
            if walls_remaining[agent_id] == 0:
                raise ValueError(f"no walls left for agent {agent_id}")
            if x == self.board_size - 1:
                raise ValueError("cannot place wall on the edge")
            elif y == self.board_size - 1:
                raise ValueError("right section out of board")
            elif np.any(board[3, x, y : y + 2]):
                raise ValueError("wall already placed")
            elif np.all(board[2, x : x + 2, y]):
                raise ValueError("cannot create intersecting walls")
            board[3, x, y] = 1
            board[3, x, y + 1] = 1
            if not self._check_path_exists(board, 0) or not self._check_path_exists(
                board, 1
            ):
                raise ValueError("cannot place wall blocking all paths")
            walls_remaining[agent_id] -= 1

        elif action_type == 3:  # Rotate section
            region_top_left = np.array([x, y])
            if not self._check_in_range(
                region_top_left,
                bottom_right=np.array([self.board_size - 3, self.board_size - 3]),
            ):
                raise ValueError("rotation region out of board")
            elif walls_remaining[agent_id] < 2:
                raise ValueError(f"less than two walls left for agent {agent_id}")

            padded_horizontal = np.pad(board[2], 1, constant_values=0)
            padded_vertical = np.pad(board[3], 1, constant_values=0)
            px, py = x + 1, y + 1
            horizontal_region = np.copy(padded_horizontal[px : px + 4, py - 1 : py + 4])
            vertical_region = np.copy(padded_vertical[px - 1 : px + 4, py : py + 4])
            horizontal_region_new = np.rot90(vertical_region)
            vertical_region_new = np.rot90(horizontal_region)
            padded_horizontal[px : px + 4, py - 1 : py + 4] = horizontal_region_new
            padded_vertical[px - 1 : px + 4, py : py + 4] = vertical_region_new
            board[2] = padded_horizontal[1:-1, 1:-1]
            board[3] = padded_vertical[1:-1, 1:-1]
            board[2, :, 8] = 0
            board[3, 8, :] = 0

            if not self._check_path_exists(board, 0) or not self._check_path_exists(
                board, 1
            ):
                raise ValueError("cannot rotate to block all paths")
            walls_remaining[agent_id] -= 2

        else:
            raise ValueError(f"invalid action_type: {action_type}")

        next_state = PuoriborState(
            board=board,
            walls_remaining=walls_remaining,
            done=self._check_wins(board),
        )
        if post_step_fn is not None:
            post_step_fn(next_state, agent_id, action)
        return next_state

    def _check_in_range(self, pos: NDArray[np.int_], bottom_right=None) -> np.bool_:
        if bottom_right is None:
            bottom_right = np.array([self.board_size, self.board_size])
        return np.all(np.logical_and(np.array([0, 0]) <= pos, pos < bottom_right))

    def _check_path_exists(self, board: NDArray[np.int_], agent_id: int) -> bool:
        start_pos = tuple(np.argwhere(board[agent_id] == 1)[0])
        visited = set()
        q = Deque([start_pos])
        goal_y = 8 if agent_id == 0 else 0
        while q:  # Run BFS to determine path
            here = q.popleft()
            if here[1] == goal_y:
                return True
            for dx, dy in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                there = (here[0] + dx, here[1] + dy)
                if not np.all(
                    np.logical_and(
                        [0, 0] <= np.array(there),
                        np.array(there) < [self.board_size, self.board_size],
                    )
                ) or self._check_wall_blocked(board, np.array(here), np.array(there)):
                    continue
                if there not in visited:
                    visited.add(there)
                    q.append(there)
        return False

    def _check_wall_blocked(
        self,
        board: NDArray[np.int_],
        current_pos: NDArray[np.int_],
        new_pos: NDArray[np.int_],
    ) -> bool:
        delta = new_pos - current_pos
        right_check = delta[0] > 0 and np.any(
            board[3, current_pos[0] : new_pos[0], current_pos[1]]
        )
        left_check = delta[0] < 0 and np.any(
            board[3, new_pos[0] : current_pos[0], current_pos[1]]
        )
        down_check = delta[1] > 0 and np.any(
            board[2, current_pos[0], current_pos[1] : new_pos[1]]
        )
        up_check = delta[1] < 0 and np.any(
            board[2, current_pos[0], new_pos[1] : current_pos[1]]
        )
        return bool(right_check or left_check or down_check or up_check)

    def _check_wins(self, board: NDArray[np.int_]) -> bool:
        return bool(board[0, :, -1].sum() or board[1, :, 0].sum())

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
            ]
        )

        initial_state = PuoriborState(
            board=starting_board,
            done=False,
            walls_remaining=np.array((self.max_walls, self.max_walls)),
        )

        return initial_state
