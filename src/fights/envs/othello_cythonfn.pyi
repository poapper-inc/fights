from typing import Tuple

import numpy as np

def fast_step(
    pre_board: np.ndarray,
    pre_legal_actions: np.ndarray,
    agent_id: int,
    action_r: int,
    action_c: int,
    board_size: int,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]: ...
