from dataclasses import dataclass
from typing import TypeAlias

import jax.numpy as jnp
from jax.scipy.signal import convolve2d

Action: TypeAlias = jnp.ndarray


@dataclass
class GomokuState:
    board: jnp.ndarray
    won: bool = False


# functional model
# deterministic execution
# no internal state
class GomokuEnv:
    win_condition: int = 3

    def step(self, state: GomokuState, action: Action) -> GomokuState:
        return GomokuState(
            board=state.board.at[action[0], action[1]].set(1),
            won=self._check_wins(state),
        )

    def _check_wins(self, state: GomokuState) -> bool:
        kernels = [
            jnp.eye(self.win_condition),
            jnp.fliplr(jnp.eye(self.win_condition)),
            jnp.ones((self.win_condition, 1)),
            jnp.ones((1, self.win_condition)),
        ]
        for kernel in kernels:
            convolved = convolve2d(state.board, kernel)
            if jnp.any(convolved >= self.win_condition):
                return True
        return False


if __name__ == "__main__":
    state = GomokuState(board=jnp.zeros((10, 10)))
    state = GomokuEnv().step(state, jnp.array([0, 1]))
    print(state.won)
    state = GomokuEnv().step(state, jnp.array([0, 2]))
    print(state.won)
    state = GomokuEnv().step(state, jnp.array([0, 3]))
    print(state.won)
