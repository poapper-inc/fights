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
            board=board, walls_remaining=np.array([10, 10])
        )
        self.env = PuoriborEnv()

    def test_action(self):
        self.assertRaisesRegex(
            ValueError,
            "invalid agent_id",
            lambda: self.env.step(self.initial_state, np.array([2, 0, 0, 0])),
        )
        self.assertRaisesRegex(
            ValueError,
            "invalid action_type",
            lambda: self.env.step(self.initial_state, np.array([0, 4, 0, 0])),
        )

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

        blocked_by_edge = deepcopy(self.initial_state)
        blocked_by_edge.board[1] = np.zeros_like(blocked_by_edge.board[1])
        blocked_by_edge.board[1, 4, 1] = 1
        expected_pos = np.zeros_like(blocked_by_edge.board[1])
        expected_pos[3, 0] = 1
        diagonal_jump = self.env.step(blocked_by_edge, np.array([1, 0, 3, 0]))
        np.testing.assert_array_equal(diagonal_jump.board[1], expected_pos)
        self.assertRaisesRegex(
            ValueError,
            "linear jump is possible",
            lambda: self.env.step(blocked_by_edge, np.array([0, 0, 3, 1])),
        )

        blocked_by_wall = deepcopy(blocked_by_edge)
        blocked_by_wall.board[2, 4, 1] = 1
        expected_pos = np.zeros_like(blocked_by_wall.board[0])
        expected_pos[5, 1] = 1
        diagonal_jump = self.env.step(blocked_by_wall, np.array([0, 0, 5, 1]))
        np.testing.assert_array_equal(diagonal_jump.board[0], expected_pos)
        blocked_by_wall.board[2, 4, 0] = 1
        self.assertRaisesRegex(
            ValueError,
            "walls",
            lambda: self.env.step(blocked_by_wall, np.array([0, 0, 3, 1])),
        )

    def test_walls(self):
        place_wall_down = self.env.step(self.initial_state, np.array([0, 1, 0, 0]))
        expected_hwall = np.zeros_like(place_wall_down.board[2])
        expected_hwall[0, 0] = 1
        expected_hwall[1, 0] = 1
        np.testing.assert_array_equal(place_wall_down.board[2], expected_hwall)
        np.testing.assert_array_equal(
            place_wall_down.walls_remaining, self.initial_state.walls_remaining - [1, 0]
        )
        self.assertRaisesRegex(
            ValueError,
            "already placed",
            lambda: self.env.step(place_wall_down, np.array([1, 1, 0, 0])),
        )
        self.assertRaisesRegex(
            ValueError,
            "intersecting walls",
            lambda: self.env.step(place_wall_down, np.array([1, 2, 0, 0])),
        )

        place_wall_right = self.env.step(self.initial_state, np.array([1, 2, 0, 0]))
        expected_vwall = np.zeros_like(place_wall_right.board[3])
        expected_vwall[0, 0] = 1
        expected_vwall[0, 1] = 1
        np.testing.assert_array_equal(place_wall_right.board[3], expected_vwall)
        np.testing.assert_array_equal(
            place_wall_right.walls_remaining,
            self.initial_state.walls_remaining - [0, 1],
        )
        self.assertRaisesRegex(
            ValueError,
            "already placed",
            lambda: self.env.step(place_wall_right, np.array([0, 2, 0, 0])),
        )

        self.assertRaisesRegex(
            ValueError,
            "edge",
            lambda: self.env.step(self.initial_state, np.array([0, 1, 0, 8])),
        )
        self.assertRaisesRegex(
            ValueError,
            "edge",
            lambda: self.env.step(self.initial_state, np.array([0, 2, 8, 0])),
        )

        self.assertRaisesRegex(
            ValueError,
            "section out",
            lambda: self.env.step(self.initial_state, np.array([0, 1, 8, 0])),
        )
        self.assertRaisesRegex(
            ValueError,
            "section out",
            lambda: self.env.step(self.initial_state, np.array([0, 2, 0, 8])),
        )

        out_of_walls = deepcopy(self.initial_state)
        out_of_walls.walls_remaining = np.zeros_like(out_of_walls.walls_remaining)
        self.assertRaisesRegex(
            ValueError,
            "no walls left",
            lambda: self.env.step(out_of_walls, np.array([0, 1, 0, 0])),
        )
        self.assertRaisesRegex(
            ValueError,
            "no walls left",
            lambda: self.env.step(out_of_walls, np.array([1, 1, 0, 0])),
        )
