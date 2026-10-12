"""Sandboxed command execution for DevMind.

Process-level isolation: allow-listed binaries, workspace-pinned paths, scrubbed
environment, rlimits (CPU / file size / address space), wall-clock timeout with
process-group kill, and bounded output. For hostile code use a container/VM too.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

MAX_OUTPUT_CHARS = 20_000
STREAM_CAP_CHARS = 60_000
SENSITIVE_ENV = re.compile(r"(API_?KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_?KEY|AUTH)", re.I)
ALLOWED_COMMANDS = {"python", "python3", "pytest", "git", "ls", "cat", "grep", "head", "tail", "find", "wc", "pwd", "echo"}
DESTRUCTIVE = {"rm", "rmdir", "mv", "dd", "mkfs", "sudo", "su", "chmod", "chown", "curl", "wget",
               "ssh", "scp", "kill", "pkill", "shutdown", "reboot", "nc", "ncat"}
GIT_AGENT_SUBCOMMANDS = {"diff", "status", "log", "show", "checkout", "restore", "stash", "branch"}
URL_RE = re.compile(r"^https://[A-Za-z0-9.-]+(:\d+)?/[A-Za-z0-9_.~/%+-]+$")
LIMITS = ('ulimit -t 120 2>/dev/null || true; ulimit -f 20000 2>/dev/null || true; '
          'ulimit -v 4000000 2>/dev/null || true; exec "$@"')


class SandboxError(Exception):
    pass


@dataclass
class ExecResult:
    returncode: int
    output: str
    truncated: bool
    duration_ms: int
    command: str
    timed_out: bool = False


def scrub_env(home: Optional[Path] = None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not SENSITIVE_ENV.search(k) and not k.startswith("PYTEST_")}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": env.get("LC_ALL", "C.UTF-8")})
    if home is not None:
        env["HOME"] = str(home)
    return env


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS):
    if len(text) <= limit:
        return text, False
    head, tail = int(limit * 0.6), int(limit * 0.4)
    return f"{text[:head]}\n...[truncated {len(text) - limit} chars]...\n{text[-tail:]}", True


class Sandbox:
    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).resolve()

    # ---------- path & command policy ----------
    def safe_path(self, p: Union[str, Path]) -> Path:
        raw = str(p)
        if "\x00" in raw:
            raise SandboxError("invalid path")
        if ".." in Path(raw).parts:
            raise SandboxError("path traversal ('..') is blocked")
        target = (Path(raw) if os.path.isabs(raw) else self.root / raw).resolve()
        try:
            target.relative_to(self.root)
        except ValueError:
            raise SandboxError(f"path escapes the workspace: {raw}")
        return target

    def _validate(self, argv: Sequence[str]):
        if not argv:
            raise SandboxError("empty command")
        exe = os.path.basename(argv[0])
        if exe in DESTRUCTIVE:
            raise SandboxError(f"blocked destructive command: {exe}")
        if exe not in ALLOWED_COMMANDS:
            raise SandboxError(f"command not allowed: {exe} (allowed: {', '.join(sorted(ALLOWED_COMMANDS))})")
        for a in argv[1:]:
            if ".." in Path(a).parts:
                raise SandboxError("path traversal ('..') is blocked")
            if os.path.isabs(a):
                self.safe_path(a)
            if "=" in a and os.path.isabs(a.split("=", 1)[1]):
                self.safe_path(a.split("=", 1)[1])
        if exe in ("python", "python3") and "-c" in argv[1:]:
            raise SandboxError("python -c is blocked; write a script file and run it")
        if exe == "find" and any(a in ("-exec", "-execdir", "-ok", "-delete") for a in argv):
            raise SandboxError("find -exec/-delete is blocked")
        if exe == "git":
            if len(argv) < 2 or argv[1].startswith("-") or argv[1] not in GIT_AGENT_SUBCOMMANDS:
                raise SandboxError(f"git subcommand not allowed (allowed: {', '.join(sorted(GIT_AGENT_SUBCOMMANDS))})")

    # ---------- execution ----------
    def run(self, command: Union[str, Sequence[str]], timeout: int = 120,
            on_output: Optional[Callable[[str], None]] = None, internal: bool = False,
            max_chars: int = MAX_OUTPUT_CHARS) -> ExecResult:
        if isinstance(command, str):
            if not internal and re.search(r"[;&|`$<>\n]", command):
                raise SandboxError("shell operators (; & | ` $ < >) are not supported; run one command at a time")
            argv = shlex.split(command)
        else:
            argv = list(command)
        if not internal:
            self._validate(argv)
        exe = os.path.basename(argv[0])
        if exe == "pytest":
            argv = [sys.executable, "-m", "pytest", *argv[1:]]
        elif exe in ("python", "python3"):
            argv = [sys.executable, *argv[1:]]
        full = ["/bin/sh", "-c", LIMITS, "sh", *argv]

        fd, path = tempfile.mkstemp(prefix="devmind_out_")
        start = time.monotonic()
        timed_out, streamed = False, 0
        try:
            with os.fdopen(fd, "wb") as w, open(path, "rb") as r:
                proc = subprocess.Popen(full, cwd=self.root, env=scrub_env(self.root), stdout=w,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
                while True:
                    try:
                        proc.wait(timeout=0.15)
                        done = True
                    except subprocess.TimeoutExpired:
                        done = False
                    if on_output and streamed < STREAM_CAP_CHARS:
                        chunk = r.read(16384).decode("utf-8", "replace")
                        if chunk:
                            streamed += len(chunk)
                            on_output(chunk)
                    if done:
                        break
                    if time.monotonic() - start > timeout:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        timed_out = True
                        break
                if on_output and streamed < STREAM_CAP_CHARS:
                    rest = r.read(STREAM_CAP_CHARS - streamed).decode("utf-8", "replace")
                    if rest:
                        on_output(rest)
            with open(path, "rb") as r2:
                text = r2.read(8_000_000).decode("utf-8", "replace")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        text, cut = truncate(text, max_chars)
        if timed_out:
            text += f"\n[timed out after {timeout}s and killed]"
        rc = 124 if timed_out else proc.returncode
        return ExecResult(rc, text, cut, int((time.monotonic() - start) * 1000), " ".join(argv), timed_out)

    # ---------- git helpers (internal, not agent-facing) ----------
    def git(self, *args: str, max_chars: int = MAX_OUTPUT_CHARS) -> ExecResult:
        base = ["git", "-c", "user.name=DevMind", "-c", "user.email=devmind@localhost", "-c", "commit.gpgsign=false"]
        return self.run([*base, *args], internal=True, timeout=60, max_chars=max_chars)

    def ensure_repo(self) -> str:
        """Make sure the workspace is a git repo and return the base commit SHA."""
        if not (self.root / ".git").exists():
            gi = self.root / ".gitignore"
            if not gi.exists():
                gi.write_text("__pycache__/\n*.pyc\n.pytest_cache/\n")
            self.git("init", "-q")
            self.git("add", "-A")
            self.git("commit", "-q", "--allow-empty", "-m", "devmind: base snapshot")
        return self.git("rev-parse", "HEAD").output.strip()


def clone_repo(url: str, dest: Union[str, Path], timeout: int = 300, allow_file: Optional[bool] = None) -> str:
    """Real shallow clone: `git clone --depth 1 <url> <dest>`."""
    if allow_file is None:
        allow_file = os.environ.get("DEVMIND_ALLOW_FILE_URLS") == "1"
    if not (URL_RE.match(url) or (allow_file and url.startswith("file:///"))):
        raise SandboxError("repo_url must be an https:// URL")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    proc = subprocess.run(["git", "clone", "--depth", "1", "--", url, str(dest)],
                          env=scrub_env(dest.parent), capture_output=True, text=True, timeout=timeout)
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        raise SandboxError(f"git clone failed: {out[-500:]}")
    return out
