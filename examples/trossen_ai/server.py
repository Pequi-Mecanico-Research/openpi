#!/usr/bin/env python3
"""
Web control panel for the Trossen AI <-> OpenPI bridge.

Serves a small dashboard (examples/trossen_ai/web/) to start/stop episodes,
change the task prompt live, watch the camera feeds, and review episode history.

"Parar" pauses the control loop in software within one control cycle (~33ms at
30Hz) and holds the arm's last commanded position; from there the operator can
either "Retomar" (resume the policy from where it paused) or "Abortar" (end the
episode and return the arm to its configured ready pose). None of this cuts
power to the motors or substitutes for a physical/hardware e-stop during live
demos.

Usage:
    uv run server.py --policy_host localhost --policy_port 8000 --mode test --web_port 8080
"""

import argparse
import asyncio
import base64
import logging
import threading
import time

import cv2
from fastapi import FastAPI
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from main import TrossenOpenPIBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WEB_DIR = "web"
CAMERA_FPS = 15
CAMERA_STREAM_WIDTH = 480  # downscaled just for the web preview; inference uses the full-res frame
CAMERA_JPEG_QUALITY = 80
WS_PUSH_HZ = 15

CONFIG = {
    "policy_host": "localhost",
    "policy_port": 8000,
    "control_freq": 30,
    "mode": "autonomous",
    "max_steps": 1000,
}

bridge: TrossenOpenPIBridge | None = None
latest_frames: dict[str, bytes] = {}
frames_lock = threading.Lock()
camera_stop_event = threading.Event()
camera_thread: threading.Thread | None = None

app = FastAPI()


class StartRequest(BaseModel):
    prompt: str
    max_steps: int | None = None


class OutcomeRequest(BaseModel):
    outcome: str


def camera_loop():
    """Continuously poll every camera directly (bypassing get_observation/the arm
    driver entirely) so the live view refreshes independently of the control loop's
    inference cadence. Frames are downscaled before encoding purely for the web
    preview; the control loop reads full-res frames separately for inference."""
    period = 1.0 / CAMERA_FPS
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, CAMERA_JPEG_QUALITY]
    while not camera_stop_event.is_set():
        for name, cam in bridge.robot.cameras.items():
            try:
                frame_bgr = cam.async_read()
                h, w = frame_bgr.shape[:2]
                if w > CAMERA_STREAM_WIDTH:
                    scale = CAMERA_STREAM_WIDTH / w
                    frame_bgr = cv2.resize(frame_bgr, (CAMERA_STREAM_WIDTH, int(h * scale)))
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                ok, buf = cv2.imencode(".jpg", frame_rgb, encode_params)
                if ok:
                    with frames_lock:
                        latest_frames[name] = buf.tobytes()
            except Exception:
                logger.exception(f"Failed to read camera '{name}'")
        camera_stop_event.wait(period)


@app.on_event("startup")
def on_startup():
    global bridge, camera_thread
    bridge = TrossenOpenPIBridge(
        policy_server_host=CONFIG["policy_host"],
        policy_server_port=CONFIG["policy_port"],
        control_frequency=CONFIG["control_freq"],
        test_mode=CONFIG["mode"],
        max_steps=CONFIG["max_steps"],
    )
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()


@app.on_event("shutdown")
def on_shutdown():
    camera_stop_event.set()
    if camera_thread is not None:
        camera_thread.join(timeout=2.0)
    if bridge is not None:
        bridge.cleanup()


@app.post("/api/start")
def start_episode(req: StartRequest):
    if bridge.is_running:
        return {"ok": False, "error": "An episode is already running"}
    thread = threading.Thread(
        target=bridge.run_episode,
        kwargs={"task_prompt": req.prompt, "max_steps": req.max_steps},
        daemon=True,
    )
    thread.start()
    return {"ok": True}


@app.post("/api/pause")
def pause_episode():
    bridge.pause_requested.set()
    return {"ok": True}


@app.post("/api/resume")
def resume_episode():
    bridge.pause_requested.clear()
    return {"ok": True}


@app.post("/api/abort")
def abort_episode():
    bridge.abort_requested.set()
    return {"ok": True}


@app.get("/api/status")
def get_status():
    return bridge.status


@app.get("/api/history")
def get_history():
    return list(reversed(bridge.history))


@app.post("/api/history/{episode_id}/outcome")
def set_outcome(episode_id: int, req: OutcomeRequest):
    for entry in bridge.history:
        if entry["id"] == episode_id:
            entry["outcome"] = req.outcome
            return {"ok": True}
    return {"ok": False, "error": "Episode not found"}


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with frames_lock:
                cameras_b64 = {name: base64.b64encode(data).decode("ascii") for name, data in latest_frames.items()}
            await websocket.send_json({"status": bridge.status, "cameras": cameras_b64})
            await asyncio.sleep(1.0 / WS_PUSH_HZ)
    except WebSocketDisconnect:
        pass


# Mounted last so it doesn't shadow the /api and /ws routes above.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main():
    parser = argparse.ArgumentParser(description="Web control panel for the Trossen AI <-> OpenPI bridge")
    parser.add_argument("--policy_host", default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz")
    parser.add_argument(
        "--mode",
        choices=["autonomous", "autonomous_base", "test"],
        default="autonomous",
        help="Operation mode: autonomous (arm only), autonomous_base (arm + mobile base) or test (no movement)",
    )
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--web_host", default="0.0.0.0", help="Host to serve the web panel on")
    parser.add_argument("--web_port", type=int, default=8080, help="Port to serve the web panel on")
    args = parser.parse_args()

    CONFIG.update(
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        control_freq=args.control_freq,
        mode=args.mode,
        max_steps=args.max_steps,
    )

    uvicorn.run(app, host=args.web_host, port=args.web_port, log_level="info")


if __name__ == "__main__":
    main()
