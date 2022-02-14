import os
import platform

from rich.console import Console  # type: ignore
from rich.table import Table  # type: ignore

from .const_variable import BOARD_LENGTH, Stone

EMPTY_STR = " "
BLACK_STR = "●"
WHITE_STR = "◯"
FORBIDDEN_STR = "X"


class TextDisplay:
    def __init__(self, board, black_player, white_player):
        if platform.system() == "Windows":
            self.__is_window = True
        else:
            self.__is_window = False
        self.__console = Console()
        self.__board = board
        self.__players = [black_player, white_player]

    def __clear(self):
        if self.__is_window:
            os.system("cls")
        else:
            os.system("clear")

    def print_board_step(self):
        self.__clear()
        print("● Player : %s" % self.__players[0])
        print("◯ Player : %s" % self.__players[1])
        str_board = self.__make_board()
        table = Table(show_header=False, show_lines=True)
        for x in range(BOARD_LENGTH):
            table.add_row(*str_board[x])
        self.__console.print(table)

    def __make_board(self):
        str_board = [[""] * BOARD_LENGTH for _ in range(BOARD_LENGTH)]
        for x in range(BOARD_LENGTH):
            for y in range(BOARD_LENGTH):
                if self.__board[x][y] == Stone.EMPTY:
                    str_board[x][y] = EMPTY_STR
                elif self.__board[x][y] == Stone.BLACK:
                    str_board[x][y] = BLACK_STR
                elif self.__board[x][y] == Stone.WHITE:
                    str_board[x][y] = WHITE_STR
                else:
                    str_board[x][y] = FORBIDDEN_STR

        return str_board
