from typing import Tuple

import numpy as np

from .quoridor import QuoridorState

def fast_step(
    pre_board: np.ndarray,
    pre_walls_remaining: np.ndarray,
    agent_id: int,
    action: np.ndarray,
    board_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]: ...
def fast_legal_actions(
    state: QuoridorState, agent_id: int, board_size: int
) -> np.ndarray: ...
