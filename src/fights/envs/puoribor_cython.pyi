from typing import Tuple

import numpy as np

from .puoribor import PuoriborState

def fast_step(
    pre_board: np.ndarray,
    pre_walls_remaining: np.ndarray,
    agent_id: int,
    action: np.ndarray,
    board_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]: ...
def legal_actions(
    state: PuoriborState, agent_id: int, board_size: int
) -> np.ndarray: ...
