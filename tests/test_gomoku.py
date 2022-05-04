# import unittest

# from fights.games.gomoku.agent import Agent
# from fights.games.gomoku.basic_rule import BasicRule
# from fights.games.gomoku.computer_agent import ComputerAgent
# from fights.games.gomoku.const_variable import (
#     COMPUTER_NAME,
#     EnvMode,
#     EnvRule,
#     Stone,
#     Turn,
# )
# from fights.games.gomoku.env_util import BOARD_LENGTH
# from fights.games.gomoku.gomoku_env import GomokuEnv
# from fights.games.gomoku.user_agent import UserAgent


# class TestInitEnv(unittest.TestCase):
#     def setUp(self):
#         self.gomoku_env = GomokuEnv()

#     def test_get_board(self):
#         self.assertTrue(self.gomoku_env.get_board())

#     def test_board_is_init(self):
#         self.assertTrue(
#             self.gomoku_env.get_board()
#             == [[Stone.EMPTY] * BOARD_LENGTH for i in range(BOARD_LENGTH)]
#         )

#     def test_board_is_not_modifiable(self):
#         self.gomoku_env.get_board().clear()
#         self.assertFalse(self.gomoku_env.get_board() == [])

#     def test_turn_is_init(self):
#         self.assertTrue(self.gomoku_env.get_turn() == Turn.WHITE)

#     def test_set_and_get_mode(self):
#         mode = EnvMode.PVE_PROGRAM
#         self.gomoku_env = GomokuEnv(env_mode=mode)
#         self.assertEqual(self.gomoku_env.get_mode(), mode)

#     def test_set_and_get_rule(self):
#         self.assertEqual(self.gomoku_env.get_rule(), EnvRule.BASIC)
#         self.gomoku_env = GomokuEnv(EnvRule.RENJU)
#         self.assertEqual(self.gomoku_env.get_rule(), EnvRule.RENJU)

#     # def test_register_agents(self):
#     #     self.assertEqual(self.gomokuEnv.get_agents_name(), ("user", "user"))


# class TestAgent(unittest.TestCase):
#     def test_set_and_get_agent_info(self):
#         name = "User1"
#         color = Stone.WHITE
#         agent = Agent(Stone.WHITE, name, True)
#         self.assertEqual(agent.get_name(), name)
#         self.assertEqual(agent.get_color(), color)


# class TestComputerAgent(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.gomoku_env = GomokuEnv()
#         cls.agent = ComputerAgent(Stone.BLACK)

#     def test_get_name(self):
#         self.assertEqual(self.agent.get_name(), COMPUTER_NAME)

#     def test_determine_action(self):
#         board = self.gomoku_env.get_board()
#         self.assertEqual(self.agent.determine_and_return_action(board), (0, 0))
#         board[0][0] = Stone.FORBIDDEN
#         board[0][1] = Stone.WHITE
#         board[0][2] = Stone.BLACK
#         self.assertEqual(self.agent.determine_and_return_action(board), (0, 3))


# class TestUserAgent(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.gomoku_env = GomokuEnv()
#         cls.agent = UserAgent(Stone.BLACK)


# # It depends on BOARD_LENGTH
# # The test was conducted assuming that BOARD_LENGTH is 15 or large.
# class TestBasicRule(unittest.TestCase):
#     def setUp(self):
#         self.board = [[Stone.EMPTY] * BOARD_LENGTH for _ in range(BOARD_LENGTH)]
#         self.rule = BasicRule(self.board)

#     def test_sync_and_check_action_validity(self):
#         self.board[0][1] = Stone.BLACK
#         self.board[0][2] = Stone.WHITE
#         self.board[0][3] = Stone.FORBIDDEN
#         self.assertFalse(self.rule.check_action_validity(0, -1))
#         self.assertFalse(self.rule.check_action_validity(BOARD_LENGTH + 1, 0))
#         self.assertFalse(self.rule.check_action_validity(0, 1))
#         self.assertFalse(self.rule.check_action_validity(0, 2))
#         self.assertFalse(self.rule.check_action_validity(0, 3))
#         self.assertTrue(self.rule.check_action_validity(0, 4))

#     def test_check_game_is_over_when_not_over(self):
#         turn = Turn.BLACK
#         self.assertFalse(self.rule.check_game_is_over(turn))
#         self.board[3][4] = Stone.WHITE
#         self.board[3][5] = Stone.BLACK
#         self.assertFalse(self.rule.check_game_is_over(turn))
#         self.board[3][6] = Stone.WHITE
#         self.board[3][7] = Stone.BLACK
#         self.board[3][8] = Stone.WHITE
#         self.assertFalse(self.rule.check_game_is_over(turn))

#     def test_check_game_is_over_when_horizontal_continue(self):
#         turn = Turn.WHITE
#         for i in range(2, 7):
#             self.board[2][i] = Stone.WHITE
#         self.assertTrue(self.rule.check_game_is_over(turn))

#     def test_check_game_is_over_when_vertical_continue(self):
#         turn = Turn.BLACK
#         for i in range(3, 8):
#             self.board[i][2] = Stone.BLACK
#         self.assertTrue(self.rule.check_game_is_over(turn))

#     def test_check_game_is_over_when_diagonal_utd_continue(self):
#         turn = Turn.WHITE
#         for i in range(5):
#             self.board[5 + i][5 + i] = Stone.WHITE
#         self.assertTrue(self.rule.check_game_is_over(turn))

#     def test_check_game_is_over_when_diagonal_dtu_continue(self):
#         turn = Turn.BLACK
#         for i in range(5):
#             self.board[10 - i][10 + i] = Stone.BLACK
#         self.assertTrue(self.rule.check_game_is_over(turn))
