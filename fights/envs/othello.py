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

from dataclasses import dataclass
import sys
import numpy as np

from typing import Callable, Optional, Dict
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

    done: bool = False
    """
    Boolean value indicating wheter the game is done.
    """

    reward: NDArray[np.int_] = np.array([0, 0])
    """
    Array of shape ''(2,)'',
    where each value indicates the reward of each agent.

    values
        - Win : 1
        - Lose : -1
        - Draw or not done yet : 0
    """

    def __str__(self) -> str:
        """
        Generate a human-readable string representation of the board.
        Uses unicode box drawing characters.
        """

        table_top = "?”Œ??????????”¬??????????”¬??????????”¬??????????”¬??????????”¬??????????”¬??????????”¬??????????”"
        vertical_wall = "?”‚"
        horizontal_wall = "?????????"
        left_intersection = "?”œ"
        middle_intersection = "?”¼"
        right_intersection = "?”¤"
        left_intersection_bottom = "?””"
        middle_intersection_bottom = "?”´"
        right_intersection_bottom = "?”˜"

        result = table_top + "\n"

        for r in range(8):
            board_line = self.board[:, r, :]
            result += vertical_wall
            for c in range(8):
                board_cell = board_line[:, c]
                if board_cell[0]:
                    result += " X "
                elif board_cell[1]:
                    result += " O "
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
                        right_intersection_bottom if c == 7
                        else middle_intersection_bottom
                    )
                else:
                    result += "   "
                    result += (
                        right_intersection if c == 7
                        else middle_intersection
                    )

            result += "\n"

        return result

    def perspective(self, agent_id: int) -> OthelloState:
        """
        Return state observed by the agent whose ID is agent_id.

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
            it returns flipped ''board'' array if ''agent_id'' is 1, and the same goes for ''legal_actions'' array.
        """

        if agent_id == 0:
            return self.board
        
        reverse_board = np.stack(
            np.fliplr(self.board[1]),
            np.fliplr(self.board[0])
        )
        reverse_legal_actions = np.stack(
            np.fliplr(self.legal_actions[1]),
            np.fliplr(self.legal_actions[0])
        )
        reverse_reward = np.fliplr(self.reward)

        reverse_state = OthelloState(
            board = reverse_board,
            legal_actions = reverse_legal_actions,
            done = self.done,
            reward = reverse_reward
        )
        
        return reverse_state

    def need_jump(self, agent_id: int) -> bool:
        """
        Return whether the agent has no legal action.
        """
        return np.count_nonzero(self.legal_actions[agent_id]) == 0

    def to_dict(self) -> Dict:
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
    env_id = ("othello", 0) # type: ignore
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
            if state.board[1-agent_id][r][c]:
                raise ValueError("cannot put a stone on opponent's stone")
            elif state.board[agent_id][r][c]:
                raise ValueError("cannot put a stone on another stone")
            else:
                raise ValueError("There is no stones to flip")

        board = np.copy(state.board)

        directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

        for dir in directions:
            stones_to_flip = []
            temp_r = r
            temp_c = c
            for _ in range(1, self.board_size):
                temp_r += dir[0]
                temp_c += dir[1]
                if not self._check_in_range(np.array([temp_r, temp_c])):
                    break
                if state.board[1-agent_id][temp_r][temp_c] == 1:
                    stones_to_flip.append((temp_r, temp_c))
                elif state.board[agent_id][temp_r][temp_c] == 1:
                    if stones_to_flip:
                        for a_stone in stones_to_flip:
                            board[1-agent_id][a_stone[0]][a_stone[1]] = 0
                            board[agent_id][a_stone[0]][a_stone[1]] = 1
                    break
                else:
                    break
        board[agent_id][r][c] = 1

        legal_actions = np.zeros((2, self.board_size, self.board_size), dtype=np.int_)
        
        for r in range(self.board_size):
            for c in range(self.board_size):

                if board[0][r][c] == 0 and board[1][r][c] == 0:

                    for agent_id in range(2):

                        flipped = False
                        for dir in directions:

                            something_to_flip = False
                            temp_r = r
                            temp_c = c
                            for _ in range(1, self.board_size):

                                temp_r += dir[0]
                                temp_c += dir[1]
                                if not self._check_in_range(np.array([temp_r, temp_c])):
                                    break
                                if board[1-agent_id][temp_r][temp_c] == 1:
                                    something_to_flip = True
                                elif board[agent_id][temp_r][temp_c] == 1:
                                    if something_to_flip:
                                        flipped = True
                                    break
                                else:
                                    break
                            if flipped:
                                break
                        
                        if flipped:
                            legal_actions[agent_id][r][c] = 1
        
        done = False
        reward = np.zeros((2,))
        if np.count_nonzero(legal_actions) == 0:
            done = True
            reward = self._check_wins(board)
        
        next_state = OthelloState(
            board = board,
            legal_actions = legal_actions,
            done = done,
            reward = reward
        )
        if post_step_fn is not None:
            post_step_fn(next_state, agent_id, action)
        return next_state

    def _check_wins(self, board: NDArray[np.int_]) -> NDArray[np.int_]:
        agent0_cnt = np.count_nonzero(board[0])
        agent1_cnt = np.count_nonzero(board[1])
        
        if agent0_cnt > agent1_cnt: return np.array([1, -1])
        elif agent0_cnt < agent1_cnt: return np.array([-1, 1])
        else: return np.array([0, 0])

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

        board = np.array([
            [[0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0]],
            [[0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0]],
        ])

        legal_actions = np.array([
            [[0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0]],
            [[0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,1,0,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0]],
        ])

        initial_state = OthelloState(
            board = board,
            legal_actions = legal_actions,
            done = False,
            reward = np.zeros((2,))
        )

        return initial_state