from .agent import Agent


class UserAgent(Agent):
    def __init__(self, color, name, by_program):
        super().__init__(color, name, by_program)

    def _determine_action_by_algorithm(self, board):
        # Where the user needs to fill it out
        pass


class UserAgent1(UserAgent):
    def __init__(self, color, name, by_program):
        super().__init__(color, name, by_program)

    def _determine_action_by_algorithm(self, board):
        # Where the user needs to fill it out
        pass


class UserAgent2(UserAgent):
    def __init__(self, color, name, by_program):
        super().__init__(color, name, by_program)

    def _determine_action_by_algorithm(self, board):
        # Where the user needs to fill it out
        pass
