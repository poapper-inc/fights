import unittest

import numpy as np

from fights.envs.puoribor import PuoriborEnv, PuoriborState


class TestPuoriborState(unittest.TestCase):
    def setUp(self):
        env = PuoriborEnv()
        self.initial_state = env.initialize_state()
        self.state = env.step(self.initial_state, 0, np.array([0, 3, 0]))
        self.state = env.step(self.state, 1, np.array([2, 3, 0]))
        self.state = env.step(self.state, 0, np.array([0, 3, 1]))

    def test_to_dict(self):
        serialized = self.state.to_dict()
        self.assertListEqual(serialized["board"], self.state.board.tolist())
        self.assertListEqual(
            serialized["walls_remaining"], self.state.walls_remaining.tolist()
        )
        self.assertEqual(serialized["done"], self.state.done)

    def test_from_dict(self):
        serialized = {
            "board": self.initial_state.board.tolist(),
            "walls_remaining": self.initial_state.walls_remaining.tolist(),
            "done": self.initial_state.done,
        }
        state = PuoriborState.from_dict(serialized)
        np.testing.assert_array_equal(state.board, self.initial_state.board)
        np.testing.assert_array_equal(
            state.walls_remaining, self.initial_state.walls_remaining
        )
        self.assertEqual(state.done, self.initial_state.done)
