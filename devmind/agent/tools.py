"""Syntax-guarded surgical editor + shell tool (the Agent-Computer Interface)."""
from __future__ import annotations

import ast

from devmind.sandbox.executor import Sandbox, SandboxError


def _fn(name, desc, props, required=None):
    params = {"type": "object", "properties": props}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}


S = {"type": "string"}
I = {"type": "integer"}
TOOL_SCHEMAS = [
    _fn("view", "View a file (numbered lines) or list a directory.",
        {"path": S, "start_line": I, "end_line": I}, ["path"]),
    _fn("str_replace", "Replace exactly one occurrence of old_str with new_str in a file.",
        {"path": S, "old_str": S, "new_str": S}, ["path", "old_str", "new_str"]),
    _fn("insert", "Insert text after the given 1-based line number (0 = top of file).",
        {"path": S, "line": I, "text": S}, ["path", "line", "text"]),
    _fn("undo_edit", "Undo the last edit made to a file.", {"path": S}, ["path"]),
    _fn("run", "Run ONE sandboxed command (python, pytest, git diff/status/log, ls, cat, grep, find...).",
        {"command": S}, ["command"]),
    _fn("finish", "Call when the issue is resolved and tests pass.", {"summary": S}, ["summary"]),
]


class Toolbox:
    def __init__(self, sandbox: Sandbox):
        self.sb = sandbox
        self.history: dict = {}

    def _rel(self, p) -> str:
        return p.relative_to(self.sb.root).as_posix()

    def checkpoint(self, message: str):
        self.sb.git("add", "-A")
        self.sb.git("commit", "-q", "--allow-empty", "-m", message)

    def view(self, path=".", start_line=None, end_line=None) -> str:
        p = self.sb.safe_path(path)
        if not p.exists():
            return f"ERROR: {path} does not exist"
        if p.is_dir():
            names = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir() if x.name != ".git")
            return "\n".join(names) or "(empty directory)"
        lines = p.read_text(errors="replace").splitlines()
        s = max(1, int(start_line or 1))
        e = min(len(lines), int(end_line or s + 199))
        body = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(s, e + 1))
        return f"{self._rel(p)} (lines {s}-{e} of {len(lines)})\n{body}"

    def write_file(self, path, new_content: str) -> str:
        """The only way edits reach disk. Pre-write AST gate -> git checkpoint -> write."""
        p = self.sb.safe_path(path)
        rel = self._rel(p)
        if p.suffix == ".py":
            try:
                ast.parse(new_content)
            except SyntaxError as e:
                return (f"AST_LINTER_ERROR: {e.msg} (line {e.lineno}, col {e.offset}). "
                        f"Edit rejected, {rel} left unchanged. Fix the syntax and retry.")
        self.checkpoint(f"checkpoint: before editing {rel}")
        old = p.read_text() if p.exists() else None
        self.history.setdefault(rel, []).append(old)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new_content)
        return f"OK: wrote {rel}"

    def str_replace(self, path, old_str, new_str) -> str:
        p = self.sb.safe_path(path)
        if not p.is_file():
            return f"ERROR: {path} is not a file"
        content = p.read_text()
        n = content.count(old_str)
        if n == 0:
            return "ERROR: old_str not found (it must match the file exactly, including whitespace)"
        if n > 1:
            return f"ERROR: old_str matches {n} locations; include more surrounding context"
        return self.write_file(path, content.replace(old_str, new_str, 1))

    def insert(self, path, line, text) -> str:
        p = self.sb.safe_path(path)
        if not p.is_file():
            return f"ERROR: {path} is not a file"
        lines = p.read_text().splitlines(keepends=True)
        line = int(line)
        if not 0 <= line <= len(lines):
            return f"ERROR: line must be between 0 and {len(lines)}"
        if lines and line == len(lines) and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.insert(line, text if text.endswith("\n") else text + "\n")
        return self.write_file(path, "".join(lines))

    def undo_edit(self, path) -> str:
        p = self.sb.safe_path(path)
        rel = self._rel(p)
        stack = self.history.get(rel)
        if not stack:
            return f"ERROR: no edits to undo for {rel}"
        old = stack.pop()
        if old is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(old)
        return f"OK: reverted last edit to {rel}"

    def run(self, command) -> str:
        res = self.sb.run(command, timeout=120)
        return f"exit code: {res.returncode}\n{res.output}"

    def dispatch(self, name: str, args: dict) -> str:
        fns = {"view": self.view, "str_replace": self.str_replace, "insert": self.insert,
               "undo_edit": self.undo_edit, "run": self.run}
        if name not in fns:
            return f"ERROR: unknown tool '{name}'"
        try:
            return fns[name](**(args or {}))
        except SandboxError as e:
            return f"SANDBOX_BLOCKED: {e}"
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}"
        except OSError as e:
            return f"ERROR: {e}"
