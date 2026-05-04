#wrapper around the 2048 game driver.
# state: 16-dimensional vector of log2(tile_value) for each cell (0 for empty).
# action: 0=Up 1=Down 2=Left 3=Right
# reward: log2(score_delta + 1) to keep values reasonable.

import asyncio
import logging
from typing import Optional

import numpy as np

from game_qa.capture.game_driver import GameDriver

logger = logging.getLogger(__name__)


class Game2048Env:
    """
    thin async RL environment.
    """

    N_ACTIONS = 4
    STATE_DIM = 16  # flattened 4×4 board, log2-encoded

    def __init__(self, driver: GameDriver):
        self.driver = driver
        self.steps = 0
        self.score = 0
        self._board_history: list[np.ndarray] = []


    def _encode(self, board: Optional[np.ndarray]) -> np.ndarray:
        """Convert raw tile values to log2-normalised floats in [0,1]."""
        if board is None:
            return np.zeros(self.STATE_DIM, dtype=np.float32)
        flat = board.flatten().astype(np.float32)
        log_vals = np.where(flat > 0, np.log2(np.maximum(flat, 1)), 0.0)
        return (log_vals / 11.0).astype(np.float32)   # 2048 = 2^11

    async def reset(self) -> np.ndarray:
        await self.driver.new_game()
        self.steps = 0
        self.score = 0
        self._board_history.clear()
        board = await self.driver.get_board_state()
        self._board_history.append(board)
        return self._encode(board)

    async def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        result = await self.driver.press(action)
        self.steps += 1

        board = await self.driver.get_board_state()
        self._board_history.append(board)

        score_delta = result["score_delta"]
        reward = np.log2(score_delta + 1) if score_delta > 0 else -0.1

        # bonus for board diversity (exploration)
        if board is not None:
            unique_tiles = len(np.unique(board[board > 0]))
            reward += unique_tiles * 0.01

        done = result["game_over"]
        info = {
            "score": result["score"],
            "action": result["action"],
            "board_changed": result["board_changed"],
            "steps": self.steps,
        }

        return self._encode(board), float(reward), done, info

    async def close(self):
        await self.driver.close()

    def board_history(self) -> list[np.ndarray]:
        return list(self._board_history)
