import asyncio
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

from game_qa.config import GAME_URL, HEADLESS, BROWSER_WIDTH, BROWSER_HEIGHT, FRAMES_DIR

logger = logging.getLogger(__name__)

ACTIONS = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]
ACTION_NAMES = ["up", "down", "left", "right"]


class GameDriver:

    def __init__(self, headless: bool = HEADLESS, save_frames: bool = True):
        self.headless = headless
        self.save_frames = save_frames
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.frame_count = 0
        self.console_logs: list[dict] = []
        self.js_errors: list[dict] = []
        self.network_errors: list[dict] = []

    async def start(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            viewport={"width": BROWSER_WIDTH, "height": BROWSER_HEIGHT}
        )
        self.page = await self._context.new_page()

        self.page.on("console", self._on_console)
        self.page.on("pageerror", self._on_page_error)
        self._context.on("requestfailed", self._on_request_failed)

        await self.page.goto(GAME_URL, wait_until="domcontentloaded", timeout=60_000)
        await asyncio.sleep(2.0)  # let JS initialise game state
        logger.info("Game loaded: %s", GAME_URL)

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    # event hooks — collect browser telemetry for log analysis     

    def _on_console(self, msg):
        entry = {"type": msg.type, "text": msg.text, "ts": time.time()}
        self.console_logs.append(entry)
        if msg.type in ("error", "warning"):
            logger.debug("[CONSOLE %s] %s", msg.type, msg.text)

    def _on_page_error(self, exc):
        entry = {"type": "pageerror", "text": str(exc), "ts": time.time()}
        self.js_errors.append(entry)
        logger.warning("[JS ERROR] %s", exc)

    def _on_request_failed(self, request):
        entry = {
            "type": "network_fail",
            "url": request.url,
            "failure": request.failure,
            "ts": time.time(),
        }
        self.network_errors.append(entry)

    # game state                                                           

    async def get_board_state(self) -> Optional[np.ndarray]: # return the 4x4 board as a numpy array of tile values (0 = empty)
        
        try:
            tiles = await self.page.evaluate("""() => {
                const cells = document.querySelectorAll('.tile');
                const board = Array(4).fill(null).map(() => Array(4).fill(0));
                cells.forEach(cell => {
                    const classes = Array.from(cell.classList);
                    const posClass = classes.find(c => c.startsWith('tile-position-'));
                    const valClass = classes.find(c => /^tile-\\d+$/.test(c));
                    if (posClass && valClass) {
                        const [col, row] = posClass.replace('tile-position-', '').split('-').map(Number);
                        const val = parseInt(valClass.replace('tile-', ''));
                        if (row >= 1 && row <= 4 && col >= 1 && col <= 4) {
                            board[row-1][col-1] = val;
                        }
                    }
                });
                return board;
            }""")
            return np.array(tiles, dtype=np.int32)
        except Exception as e:
            logger.error("get_board_state failed: %s", e)
            return None

    async def get_score(self) -> int:
        try:
            score_text = await self.page.text_content(".score-container")
            return int("".join(filter(str.isdigit, score_text or "0")) or "0")
        except Exception:
            return 0

    async def is_game_over(self) -> bool:
        try:
            over = await self.page.query_selector(".game-over")
            return over is not None
        except Exception:
            return False

    async def is_game_won(self) -> bool:
        try:
            won = await self.page.query_selector(".game-won")
            return won is not None
        except Exception:
            return False

    # actions                                                              
    async def press(self, action_idx: int) -> dict: # execute one of 4 arrow actions.

        before_score = await self.get_score()
        before_board = await self.get_board_state()

        await self.page.keyboard.press(ACTIONS[action_idx])
        await asyncio.sleep(0.15)  # let animation settle

        after_score = await self.get_score()
        after_board = await self.get_board_state()
        game_over = await self.is_game_over()

        score_delta = after_score - before_score
        board_changed = (
            before_board is not None
            and after_board is not None
            and not np.array_equal(before_board, after_board)
        )


        # returns data dict with score delta and terminal flag
        return {
            "action": ACTION_NAMES[action_idx],
            "score_delta": score_delta,
            "score": after_score,
            "board_changed": board_changed,
            "game_over": game_over,
        }

    # restart the game
    async def new_game(self):
        
        try:
            btn = await self.page.query_selector(".restart-button")
            if btn:
                await btn.click()
            else:
                await self.page.reload(wait_until="domcontentloaded")
        except Exception:
            await self.page.reload(wait_until="domcontentloaded")
        await asyncio.sleep(0.8)

   
    # capture the game container and return as H×W×3 numpy array !! dimensions are important 
    async def screenshot(self, label: str = "") -> np.ndarray:
        
        self.frame_count += 1
        try:
            container = await self.page.query_selector(".game-container")
            if container is None:
                container = self.page

            shot_bytes = await container.screenshot()
            img = Image.open(__import__("io").BytesIO(shot_bytes)).convert("RGB")

            if self.save_frames:
                fname = FRAMES_DIR / f"frame_{self.frame_count:06d}_{label}.png"
                img.save(fname)

            return np.array(img)
        except Exception as e:
            logger.error("screenshot failed: %s", e)
            return np.zeros((400, 400, 3), dtype=np.uint8)

    def get_telemetry(self) -> dict:
        return {
            "console_logs": list(self.console_logs),
            "js_errors": list(self.js_errors),
            "network_errors": list(self.network_errors),
            "frame_count": self.frame_count,
        }
