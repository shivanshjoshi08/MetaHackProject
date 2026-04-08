"""
FastAPI Backend for OpenEnv Email Triage Dashboard
===================================================
Endpoints:
  GET  /            → Health check & environment info
  GET  /health      → Smoke-test environment reset
  WS   /ws/task/{id} → Real-time agent streaming via WebSocket
"""

import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from environment import EmailTriageEnv
from baseline_inference import run_task_stream
import uvicorn

app = FastAPI(title="OpenEnv Email Triage")

# ─── CORS (allow React dev server on localhost:5173) ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "email-triage-env", "status": "ready", "tasks": [1, 2, 3]}


@app.get("/health")
def health():
    env = EmailTriageEnv()
    obs = env.reset(task_id=1)
    return {"status": "ok", "inbox_size": len(obs.inbox)}


# ─── OpenEnv Compliant REST API Endpoints ────────────────────────────────────
from pydantic import BaseModel
from typing import Dict, Any

class ResetRequest(BaseModel):
    task_id: int = 1

class StepRequest(BaseModel):
    action: Dict[str, Any]

_sessions = {}

@app.post("/reset")
def api_reset(req: ResetRequest):
    env = EmailTriageEnv()
    obs = env.reset(task_id=req.task_id)
    _sessions["default"] = env
    return obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()

@app.post("/step")
def api_step(req: StepRequest):
    env = _sessions.get("default")
    if not env:
        env = EmailTriageEnv()
        env.reset(task_id=1)
        _sessions["default"] = env
        
    from baseline_inference import parse_action
    
    try:
        act = parse_action(req.action)
    except Exception as e:
        # Standard error feedback on invalid action format
        return {"error": str(e)}

    obs, reward, done, info = env.step(act)
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs.dict(),
        "reward": float(reward.score),
        "done": done,
        "info": info
    }




@app.websocket("/ws/task/{task_id}")
async def websocket_task(websocket: WebSocket, task_id: int):
    """
    WebSocket endpoint that streams agent events in real-time.

    Protocol:
      1. Client connects to /ws/task/1 (or 2 or 3)
      2. Server streams JSON events as the agent runs
      3. When the agent finishes, the final 'complete' event is sent
      4. The connection stays open until the client disconnects

    Each event has:
      - event: str  (init | thinking | step | reward | db_result_detail | complete | error)
      - type: str   (info | action | db | dbresult | draft | reward | penalty | success)
      - text: str   (human-readable log message)
      - step: int
      - max_steps: int
      - cumulative_reward: float
      ... plus action-specific fields (category, db_result, final_score, etc.)
    """
    await websocket.accept()

    if task_id not in (1, 2, 3):
        await websocket.send_json({
            "event": "error",
            "type": "penalty",
            "text": f"❌ Invalid task_id: {task_id}. Must be 1, 2, or 3.",
            "step": 0,
            "max_steps": 0,
            "cumulative_reward": 0,
        })
        await websocket.close()
        return

    try:
        async for event in run_task_stream(task_id):
            await websocket.send_json(event)
    except WebSocketDisconnect:
        print(f"[WS] Client disconnected from task {task_id}")
    except Exception as e:
        try:
            await websocket.send_json({
                "event": "error",
                "type": "penalty",
                "text": f"❌ Server error: {str(e)}",
                "step": 0,
                "max_steps": 0,
                "cumulative_reward": 0,
            })
        except Exception:
            pass
        print(f"[WS] Error in task {task_id}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)