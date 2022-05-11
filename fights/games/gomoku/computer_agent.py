from .agent import Agent
from .const_variable import COMPUTER_NAME, Stone
from .env_util import get_board_length


class ComputerAgent(Agent):
    def __init__(self, color):
        super().__init__(color, COMPUTER_NAME, True)

    def _determine_action_by_algorithm(self, board):
        for x_i in range(get_board_length()):
            for y_i in range(get_board_length()):
                if board[x_i][y_i] == Stone.EMPTY:
                    return x_i, y_i
