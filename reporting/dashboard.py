import asyncio
import base64
import json
import logging
import threading
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from game_qa.config import BUGS_DIR, FRAMES_DIR, DASHBOARD_HOST, DASHBOARD_PORT
from game_qa.reporting.bug_report import BugReport

logger = logging.getLogger(__name__)

app = FastAPI(title="Game QA Dashboard")

# shared state updated by the orchestrator
_run_state: dict = {
    "status": "idle",
    "episode": 0,
    "step": 0,
    "score": 0,
    "epsilon": 1.0,
    "coverage": 0,
    "anomalies_detected": 0,
    "js_errors": 0,
    "hitches": 0,
    "log_analysis": {},
    "perf_summary": {},
}


def update_run_state(**kwargs):
    _run_state.update(kwargs)


@app.get("/api/state")
def get_state():
    return JSONResponse(_run_state)


@app.get("/api/bugs")
def list_bugs(limit: int = 50):
    bugs = BugReport.load_all()[:limit]
    return JSONResponse([
        {
            "id": b.id,
            "type": b.bug_type,
            "severity": b.severity,
            "score": b.composite_score,
            "description": b.description,
            "episode": b.episode,
            "step": b.step,
            "ts": b.ts,
            "screenshot": b.screenshot_path,
        }
        for b in bugs
    ])


@app.get("/api/bugs/{bug_id}/screenshot")
def get_screenshot(bug_id: str):
    for p in BUGS_DIR.glob(f"bug_{bug_id}_screenshot.png"):
        data = base64.b64encode(p.read_bytes()).decode()
        return JSONResponse({"data": f"data:image/png;base64,{data}"})
    return JSONResponse({"data": None})


@app.get("/api/bugs/{bug_id}/reconstruction")
def get_reconstruction(bug_id: str):
    for p in BUGS_DIR.glob(f"bug_{bug_id}_recon.png"):
        data = base64.b64encode(p.read_bytes()).decode()
        return JSONResponse({"data": f"data:image/png;base64,{data}"})
    return JSONResponse({"data": None})


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent.parent / "templates" / "dashboard.html"
    return HTMLResponse(html_path.read_text())

_server_thread: Optional[threading.Thread] = None


def start_dashboard():
    global _server_thread
    config = uvicorn.Config(
        app, host=DASHBOARD_HOST, port=DASHBOARD_PORT,
        log_level="warning", loop="asyncio",
    )
    server = uvicorn.Server(config)

    def _run():
        asyncio.run(server.serve())

    _server_thread = threading.Thread(target=_run, daemon=True)
    _server_thread.start()
    logger.info(
        "Dashboard running at http://%s:%d", DASHBOARD_HOST, DASHBOARD_PORT
    )
