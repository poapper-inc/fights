from enum import Enum

COMPUTER_NAME = "COM"
BOARD_LENGTH = 15


class Turn(Enum):
    BLACK = 0
    WHITE = 1


class EnvMode(Enum):
    PVE_PROGRAM = 0
    PVE_KEYBOARD = 1
    PVP_PROGRAM = 2
    PVP_KEYBOARD = 3


class EnvRule(Enum):
    BASIC = 0
    RENJU = 1


class Stone(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    FORBIDDEN = 3
