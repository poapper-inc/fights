import unittest
from copy import deepcopy

import numpy as np

from fights.envs.puoribor import PuoriborEnv, PuoriborState


class TestPuoriborEnv(unittest.TestCase):
    def setUp(self):
        initial_pos = np.zeros((9, 9), dtype=np.int8)
        initial_pos[4, 0] = 1
        board = np.array(
            [
                np.copy(initial_pos),
                np.fliplr(initial_pos),
                np.zeros((9, 9), dtype=np.int8),
                np.zeros((9, 9), dtype=np.int8),
            ]
        )
        self.initial_state = PuoriborState(
            board=board, walls_remaining=np.array([20, 20])
        )
        self.env = PuoriborEnv()

    def test_move(self):
        move_agent0_down = self.env.step(self.initial_state, np.array([0, 0, 4, 1]))
        expected_pos = np.zeros_like(self.initial_state.board[0])
        expected_pos[4, 1] = 1
        np.testing.assert_array_equal(move_agent0_down.board[0], expected_pos)

        self.assertRaisesRegex(
            ValueError,
            "out of board",
            lambda: self.env.step(self.initial_state, np.array([0, 0, 9, 9])),
        )
        self.assertRaisesRegex(
            ValueError,
            "nothing",
            lambda: self.env.step(self.initial_state, np.array([0, 0, 4, 2])),
        )
        self.assertRaisesRegex(
            ValueError,
            "diagonally",
            lambda: self.env.step(self.initial_state, np.array([0, 0, 5, 1])),
        )

        wall_placed_down = deepcopy(self.initial_state)
        wall_placed_down.board[2, 4, 0] = 1
        self.assertRaisesRegex(
            ValueError,
            "wall",
            lambda: self.env.step(wall_placed_down, np.array([0, 0, 4, 1])),
        )

        wall_placed_right = deepcopy(self.initial_state)
        wall_placed_right.board[3, 4, 0] = 1
        self.assertRaisesRegex(
            ValueError,
            "wall",
            lambda: self.env.step(wall_placed_right, np.array([0, 0, 5, 0])),
        )

        adjacent_opponent = deepcopy(self.initial_state)
        adjacent_opponent.board[1] = np.zeros_like(adjacent_opponent.board[1])
        adjacent_opponent.board[1, 4, 1] = 1
        expected_pos = np.zeros_like(adjacent_opponent.board[0])
        expected_pos[4, 2] = 1
        jump_down = self.env.step(adjacent_opponent, np.array([0, 0, 4, 2]))
        np.testing.assert_array_equal(jump_down.board[0], expected_pos)

        adjacent_opponent_with_wall = deepcopy(adjacent_opponent)
        adjacent_opponent_with_wall.board[2, 4, 1] = 1
        self.assertRaisesRegex(
            ValueError,
            "wall",
            lambda: self.env.step(adjacent_opponent_with_wall, np.array([0, 0, 4, 2])),
        )
