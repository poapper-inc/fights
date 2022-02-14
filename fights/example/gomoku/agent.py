class Agent:
    def __init__(self, color, name, by_program):
        self.__color = color
        self.__name = name
        self.__by_program = by_program
        if self.__by_program:
            self.__action_func = self._determine_action_by_algorithm
        else:
            self.__action_func = self.__determine_action_by_terminal_input

    def get_name(self):
        return self.__name

    def get_color(self):
        return self.__color

    def determine_and_return_action(self, board):
        return self.__action_func(board)

    def _determine_action_by_algorithm(self, board):
        pass

    def __determine_action_by_terminal_input(self, board):
        print("Where should you put the stone? (Ex: 1 2)")
        try:
            x, y = map(int, input().split())
        except Exception:
            print("Input format is WRONG")
            x, y = -1, -1
        return x, y
