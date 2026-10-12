"""Deterministic pipeline: LOCALIZING -> REPRODUCING -> PATCHING -> TESTING -> SELF_CORRECTING -> COMPLETE.
Every phase change is emitted only after the real work for the previous phase has finished."""
from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from devmind import config as cfg
from devmind.agent.llm import PatchContext, build_engine
from devmind.agent.repomap import build_repo_map
from devmind.agent.tools import Toolbox
from devmind.sandbox.benchmarks import TEMPLATES, materialize
from devmind.sandbox.executor import URL_RE, Sandbox, SandboxError, clone_repo

TEST_CMD = ["python", "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider"]
MAX_ROUNDS = 3
EDIT_TOOLS = {"str_replace", "insert", "undo_edit"}
TERMINAL_STATES = {"resolved", "unresolved", "failed"}
COPY_IGNORE = shutil.ignore_patterns(".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache")


class Session:
    def __init__(self, sid, title, description, source, engine, max_steps):
        self.id, self.title, self.description, self.source = sid, title, description, source
        self.engine, self.max_steps = engine, max_steps
        self.status, self.phase = "queued", "QUEUED"
        self.resolved: Optional[bool] = None
        self.error: Optional[str] = None
        self.events: list = []
        self._lock = threading.Lock()
        self.workspace: Path = cfg.WORKSPACE_ROOT / sid
        self.repomap: Optional[dict] = None
        self.diff, self.base_sha = "", None
        self.created_at, self.started_at, self.finished_at = time.time(), None, None
        self.telemetry = {"engine": engine.label, "model": engine.model, "prompt_tokens": 0,
                          "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0, "llm_latency_ms": 0,
                          "steps": 0, "max_steps": max_steps, "tests_run": 0, "elapsed_ms": 0}

    def emit(self, etype: str, **data) -> dict:
        with self._lock:
            ev = {"seq": len(self.events), "ts": time.time(), "type": etype, "phase": self.phase, **data}
            self.events.append(ev)
            return ev

    def events_after(self, seq: int) -> list:
        with self._lock:
            return self.events[seq + 1:]

    def brief(self) -> dict:
        t = dict(self.telemetry)
        if self.started_at:
            t["elapsed_ms"] = int(((self.finished_at or time.time()) - self.started_at) * 1000)
        return {"id": self.id, "title": self.title, "status": self.status, "phase": self.phase,
                "resolved": self.resolved, "error": self.error, "source": self.source,
                "max_steps": self.max_steps, "telemetry": t, "engine": self.engine.label,
                "created_at": self.created_at, "has_workspace": self.workspace.exists(),
                "has_repomap": self.repomap is not None}

    def run(self):
        Pipeline(self).run()


class Pipeline:
    def __init__(self, session: Session):
        self.s, self.engine = session, session.engine
        self.sb: Optional[Sandbox] = None
        self.ctx: Optional[PatchContext] = None
        self._phase_t: Optional[float] = None
        self.initial_rc: Optional[int] = None
        self.final_rc: Optional[int] = None
        self.skipped = False

    # ---------- helpers ----------
    def log(self, message: str):
        self.s.emit("log", message=message)

    def _phase(self, name: str):
        now = time.perf_counter()
        elapsed = int((now - self._phase_t) * 1000) if self._phase_t else 0
        prev, self._phase_t, self.s.phase = self.s.phase, now, name
        self.s.emit("phase_change", prev=prev, to=name, elapsed_ms=elapsed)

    def _tests(self, quiet: bool = False):
        s = self.s
        s.telemetry["tests_run"] += 1
        if not quiet:
            s.emit("terminal_start", cmd=" ".join(TEST_CMD))
        res = self.sb.run(TEST_CMD, timeout=180, internal=True,
                          on_output=None if quiet else (lambda c: s.emit("terminal_chunk", text=c)))
        if not quiet:
            s.emit("terminal_end", rc=res.returncode, duration_ms=res.duration_ms)
        return res.returncode, res.output

    def _update_diff(self):
        self.sb.git("add", "-A")
        d = self.sb.git("diff", "--cached", self.s.base_sha, max_chars=300_000).output
        if d != self.s.diff:
            self.s.diff = d
            self.s.emit("diff", size=len(d))

    # ---------- stages ----------
    def _prepare(self):
        s = self.s
        s.status = "preparing"
        if s.workspace.exists():
            shutil.rmtree(s.workspace)
        src = s.source
        if src["kind"] == "template":
            materialize(src["template"], s.workspace)
            self.log(f"Materialised built-in template '{src['template']}'")
        elif src["kind"] == "github":
            s.emit("terminal_start", cmd=f"git clone --depth 1 {src['url']}")
            out = clone_repo(src["url"], s.workspace)
            s.emit("terminal_chunk", text=out + "\n")
            s.emit("terminal_end", rc=0, duration_ms=0)
        else:
            shutil.copytree(src["path"], s.workspace, ignore=COPY_IGNORE)
            self.log(f"Copied local workspace {src['path']} (original left untouched)")
        self.sb = Sandbox(s.workspace)
        s.base_sha = self.sb.ensure_repo()
        s.status = "running"
        n = sum(1 for p in s.workspace.rglob("*") if p.is_file() and ".git" not in p.parts)
        self.log(f"Workspace ready at {s.workspace} ({n} files, base {s.base_sha[:8]})")

    def _localize(self):
        s = self.s
        self._phase("LOCALIZING")
        rm = build_repo_map(s.workspace, f"{s.title} {s.description}")
        s.repomap = rm
        top = [{"path": f["path"], "score": f["score"]} for f in rm["files"][:5]]
        s.emit("repomap", top=top, stats=rm["stats"])
        self.log(f"AST-parsed {rm['stats']['files']} Python files / {rm['stats']['symbols']} symbols; "
                 f"top candidates: {', '.join(t['path'] for t in top) or 'none'}")
        self.ctx = PatchContext(
            toolbox=Toolbox(self.sb), repomap=rm, issue=f"{s.title}\n\n{s.description}",
            say=self.log, run_tests=lambda quiet=True: self._tests(quiet=quiet),
            exhausted=lambda: s.telemetry["steps"] >= s.max_steps,
            record_llm=self._record_llm, step=self._step)

    def _record_llm(self, res):
        t = self.s.telemetry
        t["prompt_tokens"] += res.prompt_tokens
        t["completion_tokens"] += res.completion_tokens
        t["total_tokens"] = t["prompt_tokens"] + t["completion_tokens"]
        t["llm_calls"] += 1
        t["llm_latency_ms"] += res.latency_ms
        self.s.emit("llm", model=res.model, content=(res.content or "")[:2000],
                    prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                    latency_ms=res.latency_ms, tool_calls=[c["name"] for c in res.tool_calls])

    def _step(self, name, args, result):
        self.s.telemetry["steps"] += 1
        self.s.emit("tool", name=name, args=args, result=str(result)[:1500], step=self.s.telemetry["steps"])
        if name in EDIT_TOOLS:
            self._update_diff()

    def _reproduce(self):
        self._phase("REPRODUCING")
        self.initial_rc, out = self._tests()
        self.ctx.failure_log = out
        self.log("Bug reproduced: test-suite fails" if self.initial_rc != 0 else "Test-suite already passes")

    def _patch_loop(self):
        if self.initial_rc == 0 and not self.engine.uses_llm:
            self.skipped = True
            self.log("Nothing to repair: the offline engine needs a failing test to localise against.")
            return
        self._phase("PATCHING")
        attempted = self.engine.patch(self.ctx)
        self._update_diff()
        self._phase("TESTING")
        rc, out = self._tests()
        rounds = 0
        while rc != 0 and rounds < MAX_ROUNDS and attempted and not self.ctx.exhausted():
            rounds += 1
            self._phase("SELF_CORRECTING")
            self.log(f"Tests still failing: self-correction round {rounds}/{MAX_ROUNDS}")
            self.ctx.failure_log = out
            attempted = self.engine.patch(self.ctx)
            self._update_diff()
            self._phase("TESTING")
            rc, out = self._tests()
        self.final_rc = rc

    def _finish(self):
        s = self.s
        self._update_diff()
        self._phase("COMPLETE")
        s.resolved = bool(self.final_rc == 0 and s.diff.strip() and not self.skipped)
        s.status = "resolved" if s.resolved else "unresolved"
        msg = ("Issue resolved: tests pass with a minimal patch." if s.resolved else
               "Not resolved: no passing patch found within budget." if not self.skipped else
               "Not resolved: nothing failing to repair.")
        s.emit("result", resolved=s.resolved, message=msg, diff_bytes=len(s.diff))

    def run(self):
        s = self.s
        s.started_at = time.time()
        try:
            self._prepare()
            self._localize()
            self._reproduce()
            self._patch_loop()
            self._finish()
        except Exception as e:  # surfaced to the UI, never swallowed into fake progress
            s.error = f"{type(e).__name__}: {e}"
            s.status, s.resolved = "failed", False
            self._phase("FAILED")
            s.emit("error", message=s.error)
            try:
                if self.sb and s.base_sha:
                    self._update_diff()
            except Exception:
                pass
        finally:
            s.finished_at = time.time()
            s.telemetry["elapsed_ms"] = int((s.finished_at - s.started_at) * 1000)


class SessionManager:
    def __init__(self):
        self.sessions: dict = {}
        self._lock = threading.Lock()

    def create(self, *, title="", description="", template=None, repo_url=None, workspace_path=None,
               max_steps=None, config=None, autostart=True) -> Session:
        given = [x for x in (template, repo_url, workspace_path) if x]
        if len(given) != 1:
            raise ValueError("provide exactly one of: template, repo_url, workspace_path")
        if template:
            if template not in TEMPLATES:
                raise ValueError(f"unknown template '{template}'")
            source = {"kind": "template", "template": template}
            title = title or TEMPLATES[template]["title"]
            description = description or TEMPLATES[template]["issue"]
        elif repo_url:
            ok = URL_RE.match(repo_url) or (repo_url.startswith("file:///") and
                                            __import__("os").environ.get("DEVMIND_ALLOW_FILE_URLS") == "1")
            if not ok:
                raise ValueError("repo_url must be an https:// URL")
            source = {"kind": "github", "url": repo_url}
            title = title or repo_url.rstrip("/").rsplit("/", 1)[-1]
        else:
            p = Path(workspace_path).expanduser().resolve()
            if not p.is_dir():
                raise ValueError(f"workspace_path is not a directory: {workspace_path}")
            source = {"kind": "local", "path": str(p)}
            title = title or p.name
        engine = build_engine((config or cfg.CONFIG).snapshot())
        steps = max(1, min(int(max_steps or engine.default_max_steps), 200))
        s = Session(uuid.uuid4().hex[:8], title or "Untitled issue", description, source, engine, steps)
        with self._lock:
            self.sessions[s.id] = s
        if autostart:
            threading.Thread(target=s.run, daemon=True, name=f"devmind-{s.id}").start()
        return s

    def list(self) -> list:
        with self._lock:
            return [s.brief() for s in sorted(self.sessions.values(), key=lambda x: -x.created_at)]
