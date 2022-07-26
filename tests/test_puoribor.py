import unittest

from fights.envs.pouribor import PouriborEnv, PouriborState

from copy import deepcopy
import numpy as np


class TestPouriborEnv(unittest.TestCase):
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
        self.initial_state = PouriborState(
            board=board, walls_remaining=np.array([20, 20])
        )
        self.env = PouriborEnv()

    def test_move(self):
        move_agent0_down = self.env.step(self.initial_state, np.array([0, 0, 4, 1]))
        expected_pos = np.zeros((9, 9), dtype=np.int8)
        expected_pos[4, 1] = 1
        np.testing.assert_array_equal(move_agent0_down.board[0], expected_pos)

        self.assertRaises(
            ValueError,
            lambda: self.env.step(self.initial_state, np.array([0, 0, 9, 9])),
        )
        self.assertRaises(
            ValueError,
            lambda: self.env.step(self.initial_state, np.array([0, 0, 4, 2])),
        )

        wall_placed_down = deepcopy(self.initial_state)
        wall_placed_down.board[2, 4, 0] = 1
        self.assertRaises(
            ValueError, lambda: self.env.step(wall_placed_down, np.array([0, 0, 4, 1]))
        )

        wall_placed_right = deepcopy(self.initial_state)
        wall_placed_right.board[3, 4, 0] = 1
        self.assertRaises(
            ValueError, lambda: self.env.step(wall_placed_right, np.array([0, 0, 5, 0]))
        )
