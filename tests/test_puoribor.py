import unittest
from copy import deepcopy

import numpy as np

from fights.envs.puoribor import PuoriborEnv, PuoriborState


class TestPuoriborEnv(unittest.TestCase):
    def setUp(self):
        self.env = PuoriborEnv()
        self.initial_state = self.env.initialize_state()

    def test_initialize_state(self):
        initial_pos = np.zeros((9, 9), dtype=np.int_)
        initial_pos[4, 0] = 1
        board = np.array(
            [
                np.copy(initial_pos),
                np.fliplr(initial_pos),
                np.zeros((9, 9), dtype=np.int_),
                np.zeros((9, 9), dtype=np.int_),
            ]
        )
        initial_state_correct = PuoriborState(
            board=board, walls_remaining=np.array([10, 10])
        )

        np.testing.assert_equal(initial_state_correct.board, self.initial_state.board)
        self.assertEqual(initial_state_correct.done, self.initial_state.done)
        np.testing.assert_equal(
            initial_state_correct.walls_remaining,
            self.initial_state.walls_remaining,
        )

    def test_action(self):
        self.assertRaisesRegex(
            ValueError,
            "invalid agent_id",
            lambda: self.env.step(self.initial_state, 2, np.array([0, 0, 0])),
        )
        self.assertRaisesRegex(
            ValueError,
            "invalid action_type",
            lambda: self.env.step(self.initial_state, 0, np.array([4, 0, 0])),
        )

    def test_move(self):
        move_agent0_down = self.env.step(self.initial_state, 0, np.array([0, 4, 1]))
        expected_pos = np.zeros_like(self.initial_state.board[0])
        expected_pos[4, 1] = 1
        np.testing.assert_array_equal(move_agent0_down.board[0], expected_pos)

        self.assertRaisesRegex(
            ValueError,
            "out of board",
            lambda: self.env.step(self.initial_state, 0, np.array([0, 9, 9])),
        )
        self.assertRaisesRegex(
            ValueError,
            "nothing",
            lambda: self.env.step(self.initial_state, 0, np.array([0, 4, 2])),
        )
        self.assertRaisesRegex(
            ValueError,
            "diagonally",
            lambda: self.env.step(self.initial_state, 0, np.array([0, 5, 1])),
        )

        wall_placed_down = deepcopy(self.initial_state)
        wall_placed_down.board[2, 4, 0] = 1
        self.assertRaisesRegex(
            ValueError,
            "wall",
            lambda: self.env.step(wall_placed_down, 0, np.array([0, 4, 1])),
        )

        wall_placed_right = deepcopy(self.initial_state)
        wall_placed_right.board[3, 4, 0] = 1
        self.assertRaisesRegex(
            ValueError,
            "wall",
            lambda: self.env.step(wall_placed_right, 0, np.array([0, 5, 0])),
        )

        adjacent_opponent = deepcopy(self.initial_state)
        adjacent_opponent.board[1] = np.zeros_like(adjacent_opponent.board[1])
        adjacent_opponent.board[1, 4, 1] = 1
        expected_pos = np.zeros_like(adjacent_opponent.board[0])
        expected_pos[4, 2] = 1
        jump_down = self.env.step(adjacent_opponent, 0, np.array([0, 4, 2]))
        np.testing.assert_array_equal(jump_down.board[0], expected_pos)

        adjacent_opponent_with_wall = deepcopy(adjacent_opponent)
        adjacent_opponent_with_wall.board[2, 4, 1] = 1
        self.assertRaisesRegex(
            ValueError,
            "wall",
            lambda: self.env.step(adjacent_opponent_with_wall, 0, np.array([0, 4, 2])),
        )

        blocked_by_edge = deepcopy(self.initial_state)
        blocked_by_edge.board[1] = np.zeros_like(blocked_by_edge.board[1])
        blocked_by_edge.board[1, 4, 1] = 1
        expected_pos = np.zeros_like(blocked_by_edge.board[1])
        expected_pos[3, 0] = 1
        diagonal_jump = self.env.step(blocked_by_edge, 1, np.array([0, 3, 0]))
        np.testing.assert_array_equal(diagonal_jump.board[1], expected_pos)
        self.assertRaisesRegex(
            ValueError,
            "linear jump is possible",
            lambda: self.env.step(blocked_by_edge, 0, np.array([0, 3, 1])),
        )

        blocked_by_wall = deepcopy(blocked_by_edge)
        blocked_by_wall.board[2, 4, 1] = 1
        expected_pos = np.zeros_like(blocked_by_wall.board[0])
        expected_pos[5, 1] = 1
        diagonal_jump = self.env.step(blocked_by_wall, 0, np.array([0, 5, 1]))
        np.testing.assert_array_equal(diagonal_jump.board[0], expected_pos)
        blocked_by_wall.board[2, 4, 0] = 1
        self.assertRaisesRegex(
            ValueError,
            "walls",
            lambda: self.env.step(blocked_by_wall, 0, np.array([0, 3, 1])),
        )

    def test_walls(self):
        place_wall_down = self.env.step(self.initial_state, 0, np.array([1, 0, 0]))
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
            lambda: self.env.step(place_wall_down, 1, np.array([1, 0, 0])),
        )
        self.assertRaisesRegex(
            ValueError,
            "intersecting walls",
            lambda: self.env.step(place_wall_down, 1, np.array([2, 0, 0])),
        )

        place_wall_right = self.env.step(self.initial_state, 1, np.array([2, 0, 0]))
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
            lambda: self.env.step(place_wall_right, 0, np.array([2, 0, 0])),
        )

        self.assertRaisesRegex(
            ValueError,
            "edge",
            lambda: self.env.step(self.initial_state, 0, np.array([1, 0, 8])),
        )
        self.assertRaisesRegex(
            ValueError,
            "edge",
            lambda: self.env.step(self.initial_state, 0, np.array([2, 8, 0])),
        )

        self.assertRaisesRegex(
            ValueError,
            "section out",
            lambda: self.env.step(self.initial_state, 0, np.array([1, 8, 0])),
        )
        self.assertRaisesRegex(
            ValueError,
            "section out",
            lambda: self.env.step(self.initial_state, 0, np.array([2, 0, 8])),
        )

        out_of_walls = deepcopy(self.initial_state)
        out_of_walls.walls_remaining = np.zeros_like(out_of_walls.walls_remaining)
        self.assertRaisesRegex(
            ValueError,
            "no walls left",
            lambda: self.env.step(out_of_walls, 0, np.array([1, 0, 0])),
        )
        self.assertRaisesRegex(
            ValueError,
            "no walls left",
            lambda: self.env.step(out_of_walls, 1, np.array([1, 0, 0])),
        )

        block_path = self.env.step(self.initial_state, 0, np.array([1, 4, 0]))
        block_path = self.env.step(block_path, 1, np.array([2, 5, 0]))
        self.assertRaisesRegex(
            ValueError,
            "blocking all paths",
            lambda: self.env.step(block_path, 0, np.array([2, 3, 0])),
        )

        block_path = self.env.step(self.initial_state, 0, np.array([2, 5, 0]))
        block_path = self.env.step(block_path, 1, np.array([2, 3, 0]))
        self.assertRaisesRegex(
            ValueError,
            "blocking all paths",
            lambda: self.env.step(block_path, 0, np.array([1, 4, 0])),
        )

    def test_rotate(self):
        top_left_corner = self.env.step(self.initial_state, 0, np.array([1, 0, 0]))
        top_left_corner = self.env.step(top_left_corner, 1, np.array([1, 3, 2]))
        top_left_corner = self.env.step(top_left_corner, 0, np.array([2, 2, 0]))
        top_left_corner = self.env.step(top_left_corner, 1, np.array([2, 1, 2]))
        top_left_corner = self.env.step(top_left_corner, 0, np.array([3, 0, 0]))
        expected_hwall = np.zeros_like(top_left_corner.board[2])
        expected_hwall[:2, 1] = 1
        expected_hwall[2:5, 2] = 1
        np.testing.assert_array_equal(top_left_corner.board[2], expected_hwall)
        expected_vwall = np.zeros_like(top_left_corner.board[3])
        expected_vwall[2, :2] = 1
        expected_vwall[0, 3] = 1
        np.testing.assert_array_equal(top_left_corner.board[3], expected_vwall)

        left_edge = self.env.step(self.initial_state, 0, np.array([1, 0, 1]))
        left_edge = self.env.step(left_edge, 1, np.array([1, 3, 3]))
        left_edge = self.env.step(left_edge, 0, np.array([2, 2, 1]))
        left_edge = self.env.step(left_edge, 1, np.array([2, 1, 4]))
        left_edge = self.env.step(left_edge, 0, np.array([3, 0, 1]))
        expected_hwall = np.zeros_like(left_edge.board[2])
        expected_hwall[0, 2] = 1
        expected_hwall[2:5, 3] = 1
        np.testing.assert_array_equal(left_edge.board[2], expected_hwall)
        expected_vwall = np.zeros_like(left_edge.board[3])
        expected_vwall[2, 1:3] = 1
        expected_vwall[0, 4] = 1
        expected_vwall[1, 5] = 1
        np.testing.assert_array_equal(left_edge.board[3], expected_vwall)

        top_edge = self.env.step(self.initial_state, 0, np.array([1, 2, 3]))
        top_edge = self.env.step(top_edge, 1, np.array([1, 5, 1]))
        top_edge = self.env.step(top_edge, 0, np.array([2, 3, 3]))
        top_edge = self.env.step(top_edge, 1, np.array([2, 5, 0]))
        top_edge = self.env.step(top_edge, 0, np.array([2, 1, 0]))
        top_edge = self.env.step(top_edge, 1, np.array([3, 2, 0]))
        expected_hwall = np.zeros_like(top_edge.board[2])
        expected_hwall[2, 1] = 1
        expected_hwall[6, 1] = 1
        expected_hwall[4:6, 3] = 1
        np.testing.assert_array_equal(top_edge.board[2], expected_hwall)
        expected_vwall = np.zeros_like(top_edge.board[3])
        expected_vwall[1, :2] = 1
        expected_vwall[3, 3:5] = 1
        np.testing.assert_array_equal(top_edge.board[3], expected_vwall)

        contained = self.env.step(self.initial_state, 0, np.array([1, 2, 4]))
        contained = self.env.step(contained, 1, np.array([1, 5, 1]))
        contained = self.env.step(contained, 0, np.array([2, 1, 1]))
        contained = self.env.step(contained, 1, np.array([2, 3, 3]))
        contained = self.env.step(contained, 0, np.array([2, 4, 0]))
        contained = self.env.step(contained, 1, np.array([3, 2, 1]))
        expected_hwall = np.zeros_like(contained.board[2])
        expected_hwall[4:6, 0] = 1
        expected_hwall[2:4, 2] = 1
        expected_hwall[5, 3] = 1
        expected_hwall[6, 1] = 1
        np.testing.assert_array_equal(contained.board[2], expected_hwall)
        expected_vwall = np.zeros_like(contained.board[3])
        expected_vwall[1, 1:3] = 1
        expected_vwall[4, 0] = 1
        expected_vwall[4, 4] = 1
        np.testing.assert_array_equal(contained.board[3], expected_vwall)

        removed_bottom = self.env.step(self.initial_state, 0, np.array([2, 3, 6]))
        removed_bottom = self.env.step(removed_bottom, 1, np.array([3, 0, 5]))
        expected_hwall = np.zeros_like(removed_bottom.board[2])
        np.testing.assert_array_equal(removed_bottom.board[2], expected_hwall)
        expected_vwall = np.zeros_like(removed_bottom.board[3])
        np.testing.assert_array_equal(removed_bottom.board[3], expected_vwall)

        removed_right = self.env.step(self.initial_state, 0, np.array([1, 7, 0]))
        removed_right = self.env.step(removed_right, 1, np.array([3, 5, 1]))
        expected_hwall = np.zeros_like(removed_right.board[2])
        np.testing.assert_array_equal(removed_right.board[2], expected_hwall)
        expected_vwall = np.zeros_like(removed_right.board[3])
        np.testing.assert_array_equal(removed_right.board[3], expected_vwall)

        self.assertRaisesRegex(
            ValueError,
            "region out of board",
            lambda: self.env.step(self.initial_state, 0, np.array([3, 6, 0])),
        )

        almost_block_agent0 = self.env.step(self.initial_state, 0, np.array([2, 3, 0]))
        almost_block_agent0 = self.env.step(almost_block_agent0, 1, np.array([2, 4, 0]))
        almost_block_agent0 = self.env.step(almost_block_agent0, 0, np.array([2, 0, 1]))
        self.assertRaisesRegex(
            ValueError,
            "rotate to block",
            lambda: self.env.step(almost_block_agent0, 1, np.array([3, 1, 2])),
        )

        almost_block_agent1 = self.env.step(self.initial_state, 0, np.array([1, 5, 5]))
        almost_block_agent1 = self.env.step(almost_block_agent1, 1, np.array([1, 5, 6]))
        almost_block_agent1 = self.env.step(almost_block_agent1, 0, np.array([2, 4, 6]))
        self.assertRaisesRegex(
            ValueError,
            "rotate to block",
            lambda: self.env.step(almost_block_agent1, 1, np.array([3, 2, 5])),
        )

        lacking_walls = deepcopy(self.initial_state)
        lacking_walls.walls_remaining[0] = 1
        self.assertRaisesRegex(
            ValueError,
            "less than two walls",
            lambda: self.env.step(lacking_walls, 0, np.array([3, 0, 0])),
        )
        lacking_walls.walls_remaining[0] = 0
        self.assertRaisesRegex(
            ValueError,
            "less than two walls",
            lambda: self.env.step(lacking_walls, 0, np.array([3, 0, 0])),
        )

    def test_step_callback(self):
        class StepLogger:
            log = []

            def __call__(self, state, agent_id, action):
                self.log.append((state, agent_id, action))

        logger = StepLogger()
        action = np.array([0, 3, 0])
        next_state = self.env.step(
            self.initial_state,
            0,
            action,
            pre_step_fn=logger,
            post_step_fn=logger,
        )
        np.testing.assert_array_equal(logger.log[0][0].board, self.initial_state.board)
        self.assertEqual(logger.log[0][1], 0)
        np.testing.assert_array_equal(logger.log[0][2], action)
        np.testing.assert_array_equal(logger.log[1][0].board, next_state.board)
        self.assertEqual(logger.log[1][1], 0)
        np.testing.assert_array_equal(logger.log[1][2], action)


if __name__ == "__main__":
    unittest.main()
