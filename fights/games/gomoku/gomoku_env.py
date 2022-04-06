from abc import ABC

from ...base import BaseEnv


class GomokuEnv(BaseEnv):
    def __init__(self):
        super().__init__()

    def step(self, action):
        pass

    def reset(self):
        pass

    def observe(self, agent):
        pass

    def observation_space(self, agent):
        pass

    def action_space(self, agent):
        pass

# from .basic_rule import BasicRule
# from .computer_agent import ComputerAgent
# from .const_variable import BOARD_LENGTH, EnvMode, EnvRule, Stone, Turn
# from .renju_rule import RenjuRule
# from .text_display import TextDisplay
# from .user_agent import UserAgent1
#
#
# class GomokuEnv:
#     def __init__(self, env_rule=EnvRule.BASIC, env_mode=EnvMode.PVP_KEYBOARD):
#         self.__turn = Turn.WHITE
#         self.__init_board()
#         self.__set_rule(env_rule)
#         self.__set_mode(env_mode)
#         self.__display = TextDisplay(
#             self.__board, self.__agents[0].get_name(), self.__agents[1].get_name()
#         )
#
#     def get_board(self):
#         # deepcopy module is slow
#         board_copy = [row[:] for row in self.__board]
#         return board_copy
#
#     def __init_board(self):
#         self.__board = [[Stone.EMPTY] * BOARD_LENGTH for _ in range(BOARD_LENGTH)]
#
#     def get_turn(self):
#         return self.__turn
#
#     def get_mode(self):
#         return self.__env_mode_enum
#
#     def get_rule(self):
#         return self.__env_rule_enum
#
#     def __set_mode(self, env_mode):
#         self.__env_mode_enum = env_mode
#         if self.__env_mode_enum == EnvMode.PVE_PROGRAM:
#             self.__agents = (
#                 UserAgent1(Stone.BLACK, "User1", True),
#                 ComputerAgent(Stone.WHITE),
#             )
#         elif self.__env_mode_enum == EnvMode.PVP_PROGRAM:
#             self.__agents = (
#                 UserAgent1(Stone.BLACK, "User1", True),
#                 UserAgent1(Stone.WHITE, "User2", True),
#             )
#         elif self.__env_mode_enum == EnvMode.PVE_KEYBOARD:
#             self.__agents = (
#                 UserAgent1(Stone.BLACK, "User1", False),
#                 ComputerAgent(Stone.WHITE),
#             )
#         elif self.__env_mode_enum == EnvMode.PVP_KEYBOARD:
#             self.__agents = (
#                 UserAgent1(Stone.BLACK, "User1", False),
#                 UserAgent1(Stone.WHITE, "User2", False),
#             )
#
#     def __set_rule(self, env_rule):
#         self.__env_rule_enum = env_rule
#         if self.__env_rule_enum == EnvRule.BASIC:
#             self.__env_rule = BasicRule(self.__board)
#         elif self.__env_rule_enum == EnvRule.RENJU:
#             self.__env_rule = RenjuRule(self.__board)
#         else:
#             self.__env_rule = BasicRule(self.__board)
#
#     def next_turn(self):
#         self.__turn = Turn((self.__turn.value + 1) % 2)
#
#     def print_winner(self):
#         agent = self.__agents[self.__turn.value]
#         self.__display.print_board_step()
#         print("%s is Win!" % (agent.get_name()))
#
#     def run(self):
#         while not self.__env_rule.check_game_is_over(self.__turn):
#             self.next_turn()
#             self.__env_rule.set_board_forbidden_cell()
#             agent = self.__agents[self.__turn.value]
#             self.__display.print_board_step()
#             print("It's %s's turn." % agent.get_name())
#             x, y = agent.determine_and_return_action(board=self.get_board())
#             if not self.__env_rule.check_action_validity(x, y):
#                 pass
#             else:
#                 self.__board[x][y] = agent.get_color()
#         self.print_winner()
