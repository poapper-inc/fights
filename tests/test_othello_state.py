import unittest

import numpy as np

from fights.envs.othello import OthelloEnv, OthelloState


class TestOthelloState(unittest.TestCase):
    def setUp(self) -> None:
        self.env = OthelloEnv()
        self.initial_state = self.env.initialize_state()

    def test_serialization(self):
        serialized = self.initial_state.to_dict()
        deserialized = OthelloState.from_dict(serialized)
        np.testing.assert_array_equal(self.initial_state.board, deserialized.board)
        np.testing.assert_array_equal(
            self.initial_state.legal_actions, deserialized.legal_actions
        )
        np.testing.assert_array_equal(self.initial_state.reward, deserialized.reward)
        self.assertEqual(self.initial_state.done, deserialized.done)

    def test_perspective(self):
        before_rotation = self.env.step(self.initial_state, 0, [2, 3])
        np.testing.assert_array_equal(
            before_rotation.board, before_rotation.perspective(0)
        )
        rotated_board = before_rotation.perspective(1)
        np.testing.assert_array_equal(
            np.flip(np.rot90(before_rotation.board, 2, axes=(1, 2)), axis=0),
            rotated_board,
        )


if __name__ == "__main__":
    unittest.main()
