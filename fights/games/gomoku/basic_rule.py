from .const_variable import BOARD_LENGTH, Stone, Turn


class BasicRule:
    def __init__(self, board):
        self.__board = board

    def check_action_validity(self, x, y):
        validity = True
        if x >= BOARD_LENGTH or x < 0:
            validity = False
        elif y >= BOARD_LENGTH or y < 0:
            validity = False
        elif self.__board[x][y] != Stone.EMPTY:
            validity = False
        return validity

    def set_board_forbidden_cell(self):
        pass

    def check_game_is_over(self, turn):
        color = Stone.BLACK
        if turn == Turn.BLACK:
            color = Stone.BLACK
        if turn == Turn.WHITE:
            color = Stone.WHITE
        game_is_over = False
        if self.__check_horizontal_continue(color):
            game_is_over = True
        elif self.__check_vertical_continue(color):
            game_is_over = True
        elif self.__check_diagonal_utd_continue(color):
            game_is_over = True
        elif self.__check_diagonal_dtu_continue(color):
            game_is_over = True
        return game_is_over

    def __check_horizontal_continue(self, color):
        for x in range(BOARD_LENGTH):
            continue_count = 0
            for y in range(BOARD_LENGTH):
                if self.__board[x][y] == color:
                    continue_count += 1
                else:
                    continue_count = 0
                if continue_count == 5:
                    return True
        return False

    def __check_vertical_continue(self, color):
        for y in range(BOARD_LENGTH):
            continue_count = 0
            for x in range(BOARD_LENGTH):
                if self.__board[x][y] == color:
                    continue_count += 1
                else:
                    continue_count = 0
                if continue_count == 5:
                    return True
        return False

    # utd : Up To Down
    # It means the direction in which x increases on board[x][y]
    # when following a diagonal line from left to right
    def __check_diagonal_utd_continue(self, color):
        for i in range(BOARD_LENGTH - 4):
            continue_count = 0
            for j in range(BOARD_LENGTH - i):
                if self.__board[i + j][j] == color:
                    continue_count += 1
                else:
                    continue_count = 0
                if continue_count == 5:
                    return True

        for i in range(BOARD_LENGTH - 4):
            continue_count = 0
            for j in range(BOARD_LENGTH - i):
                if self.__board[j][i + j] == color:
                    continue_count += 1
                else:
                    continue_count = 0
                if continue_count == 5:
                    return True
        return False

    def __check_diagonal_dtu_continue(self, color):
        for i in range(4, BOARD_LENGTH):
            continue_count = 0
            for j in range(i + 1):
                if self.__board[i - j][j] == color:
                    continue_count += 1
                else:
                    continue_count = 0
                if continue_count == 5:
                    return True

        for i in range(BOARD_LENGTH - 5, -1, -1):
            continue_count = 0
            for j in range(BOARD_LENGTH - i):
                if self.__board[i + j][BOARD_LENGTH - j - 1] == color:
                    continue_count += 1
                else:
                    continue_count = 0
                if continue_count == 5:
                    return True
        return False
