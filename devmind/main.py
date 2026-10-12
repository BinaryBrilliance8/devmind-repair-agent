"""FastAPI app: REST + WebSocket telemetry + static dashboard."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from devmind import config as cfg
from devmind.agent.core import TERMINAL_STATES, SessionManager
from devmind.agent.repomap import build_repo_map
from devmind.sandbox.benchmarks import TEMPLATES
from devmind.sandbox.executor import Sandbox, SandboxError

app = FastAPI(title="DevMind", version="0.1.0")
MANAGER = SessionManager()
STATIC = Path(__file__).parent / "static"


class IssueIn(BaseModel):
    title: str = ""
    description: str = ""
    template: Optional[str] = None
    repo_url: Optional[str] = None
    workspace_path: Optional[str] = None
    max_steps: Optional[int] = None


class ConfigIn(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


def _get(sid: str):
    s = MANAGER.sessions.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    return s


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def get_config():
    return cfg.CONFIG.public()


@app.post("/api/config")
def set_config(body: ConfigIn):
    try:
        cfg.CONFIG.update(**body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return cfg.CONFIG.public()


@app.get("/api/templates")
def templates():
    return [{"id": k, "title": v["title"], "issue": v["issue"]} for k, v in TEMPLATES.items()]


@app.get("/api/sessions")
def sessions():
    return MANAGER.list()


@app.post("/api/issues")
def create_issue(body: IssueIn):
    try:
        s = MANAGER.create(**body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return s.brief()


@app.post("/api/benchmarks/run")
def run_benchmarks():
    return [MANAGER.create(title=t["title"], description=t["issue"], template=tid).brief()
            for tid, t in TEMPLATES.items()]


@app.get("/api/sessions/{sid}")
def session(sid: str):
    s = _get(sid)
    return {**s.brief(), "description": s.description, "diff": s.diff}


@app.get("/api/sessions/{sid}/diff")
def diff(sid: str):
    return {"diff": _get(sid).diff}


@app.get("/api/sessions/{sid}/repomap")
def repomap(sid: str):
    s = _get(sid)
    if s.repomap is None:
        if not s.workspace.exists():
            raise HTTPException(404, "No workspace loaded yet")
        s.repomap = build_repo_map(s.workspace, f"{s.title} {s.description}")
    return s.repomap


def _tree(root: Path, base: Path, depth=0) -> list:
    out = []
    for p in sorted(base.iterdir(), key=lambda x: (x.is_file(), x.name)):
        if p.name in {".git", "__pycache__", ".pytest_cache", "node_modules"} or depth > 6:
            continue
        rel = p.relative_to(root).as_posix()
        out.append({"name": p.name, "path": rel, "type": "dir", "children": _tree(root, p, depth + 1)}
                   if p.is_dir() else {"name": p.name, "path": rel, "type": "file"})
    return out


@app.get("/api/sessions/{sid}/files")
def files(sid: str):
    s = _get(sid)
    if not s.workspace.exists():
        raise HTTPException(404, "No workspace loaded yet")
    return _tree(s.workspace, s.workspace)


@app.get("/api/sessions/{sid}/file")
def file(sid: str, path: str):
    s = _get(sid)
    if not s.workspace.exists():
        raise HTTPException(404, "No workspace loaded yet")
    try:
        p = Sandbox(s.workspace).safe_path(path)
    except SandboxError as e:
        raise HTTPException(400, str(e))
    if not p.is_file():
        raise HTTPException(404, "not a file")
    data = p.read_bytes()[:400_000]
    return {"path": path, "content": data.decode("utf-8", "replace")}


@app.websocket("/ws/sessions/{sid}")
async def ws(sock: WebSocket, sid: str, after: int = -1):
    await sock.accept()
    s = MANAGER.sessions.get(sid)
    if not s:
        await sock.close(code=4404)
        return
    last, last_diff, first = after, None, True
    try:
        while True:
            evs = s.events_after(last)
            diff_changed = s.diff != last_diff
            if evs or diff_changed or first:
                payload = {"events": evs, "state": s.brief()}
                if diff_changed:
                    payload["diff"], last_diff = s.diff, s.diff
                await sock.send_json(payload)
                if evs:
                    last = evs[-1]["seq"]
                first = False
            await asyncio.sleep(0.15)
    except (WebSocketDisconnect, RuntimeError):
        return


app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
