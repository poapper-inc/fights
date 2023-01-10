import unittest
from copy import deepcopy

import numpy as np

from fights.envs.quoridor import QuoridorEnv


class TestQuoridorEnv(unittest.TestCase):
    def setUp(self):
        self.env = QuoridorEnv()
        self.initial_state = self.env.initialize_state()

    def test_action(self):
        self.assertRaisesRegex(
            ValueError,
            "invalid agent_id",
            lambda: self.env.step(self.initial_state, 2, [0, 0, 0]),
        )
        self.assertRaisesRegex(
            ValueError,
            "invalid action_type",
            lambda: self.env.step(self.initial_state, 0, [4, 0, 0]),
        )
        self.assertRaisesRegex(
            ValueError,
            "invalid action_type",
            lambda: self.env.step(self.initial_state, 0, [3, 0, 0]),
        )

    def test_move(self):
        move_agent0_down = self.env.step(self.initial_state, 0, [0, 4, 1])
        expected_pos = np.zeros_like(self.initial_state.board[0])
        expected_pos[4, 1] = 1
        np.testing.assert_array_equal(move_agent0_down.board[0], expected_pos)

        self.assertRaisesRegex(
            ValueError,
            "out of board",
            lambda: self.env.step(self.initial_state, 0, [0, 9, 9]),
        )
        self.assertRaisesRegex(
            ValueError,
            "nothing",
            lambda: self.env.step(self.initial_state, 0, [0, 4, 2]),
        )
        self.assertRaisesRegex(
            ValueError,
            "diagonally",
            lambda: self.env.step(self.initial_state, 0, [0, 5, 1]),
        )

        wall_placed_down = deepcopy(self.initial_state)
        wall_placed_down.board[2, 4, 0] = 1
        self.assertRaisesRegex(
            ValueError,
            "wall",
            lambda: self.env.step(wall_placed_down, 0, [0, 4, 1]),
        )

        wall_placed_right = deepcopy(self.initial_state)
        wall_placed_right.board[3, 4, 0] = 1
        self.assertRaisesRegex(
            ValueError,
            "wall",
            lambda: self.env.step(wall_placed_right, 0, [0, 5, 0]),
        )

    def test_walls(self):
        place_wall_down = self.env.step(self.initial_state, 0, [1, 0, 0])
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
            lambda: self.env.step(place_wall_down, 1, [1, 0, 0]),
        )
        self.assertRaisesRegex(
            ValueError,
            "intersecting walls",
            lambda: self.env.step(place_wall_down, 1, [2, 0, 0]),
        )

        place_wall_right = self.env.step(self.initial_state, 1, [2, 0, 0])
        expected_vwall = np.zeros_like(place_wall_right.board[3])
        expected_vwall[0, 0] = 2
        expected_vwall[0, 1] = 2
        np.testing.assert_array_equal(place_wall_right.board[3], expected_vwall)
        np.testing.assert_array_equal(
            place_wall_right.walls_remaining,
            self.initial_state.walls_remaining - [0, 1],
        )
        self.assertRaisesRegex(
            ValueError,
            "already placed",
            lambda: self.env.step(place_wall_right, 0, [2, 0, 0]),
        )

        self.assertRaisesRegex(
            ValueError,
            "edge",
            lambda: self.env.step(self.initial_state, 0, [1, 0, 8]),
        )
        self.assertRaisesRegex(
            ValueError,
            "edge",
            lambda: self.env.step(self.initial_state, 0, [2, 8, 0]),
        )

        self.assertRaisesRegex(
            ValueError,
            "section out",
            lambda: self.env.step(self.initial_state, 0, [1, 8, 0]),
        )
        self.assertRaisesRegex(
            ValueError,
            "section out",
            lambda: self.env.step(self.initial_state, 0, [2, 0, 8]),
        )

        out_of_walls = deepcopy(self.initial_state)
        out_of_walls.walls_remaining = np.zeros_like(out_of_walls.walls_remaining)
        self.assertRaisesRegex(
            ValueError,
            "no walls left",
            lambda: self.env.step(out_of_walls, 0, [1, 0, 0]),
        )
        self.assertRaisesRegex(
            ValueError,
            "no walls left",
            lambda: self.env.step(out_of_walls, 1, [1, 0, 0]),
        )

        block_path = self.env.step(self.initial_state, 0, [1, 4, 0])
        block_path = self.env.step(block_path, 1, [2, 5, 0])
        self.assertRaisesRegex(
            ValueError,
            "blocking all paths",
            lambda: self.env.step(block_path, 0, [2, 3, 0]),
        )

        block_path = self.env.step(self.initial_state, 0, [2, 5, 0])
        block_path = self.env.step(block_path, 1, [2, 3, 0])
        self.assertRaisesRegex(
            ValueError,
            "blocking all paths",
            lambda: self.env.step(block_path, 0, [1, 4, 0]),
        )

        issue_24 = self.env.step(self.initial_state, 0, [1, 2, 2])
        issue_24 = self.env.step(issue_24, 1, [1, 4, 2])
        issue_24 = self.env.step(issue_24, 0, [2, 3, 2])
        expected_hwall = np.zeros_like(issue_24.board[2])
        expected_vwall = np.zeros_like(issue_24.board[3])
        expected_hwall[2:4, 2] = 1
        expected_hwall[4:6, 2] = 2
        expected_vwall[3, 2:4] = 1
        np.testing.assert_array_equal(issue_24.board[2], expected_hwall)
        np.testing.assert_array_equal(issue_24.board[3], expected_vwall)


if __name__ == "__main__":
    unittest.main()
