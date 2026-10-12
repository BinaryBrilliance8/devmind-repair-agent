"""LLM routing (OpenAI-compatible: Gemini / Groq) + the DynamicSmartOfflineEngine."""
from __future__ import annotations

import io
import json
import re
import time
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from devmind.agent.repomap import is_test_path, render_repo_map_text
from devmind.agent.tools import TOOL_SCHEMAS


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    content: str = ""
    tool_calls: list = field(default_factory=list)       # [{"id","name","arguments"}]
    raw_tool_calls: list = field(default_factory=list)   # OpenAI wire format, echoed back
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    model: str = ""


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120):
        self.base_url, self.api_key, self.model, self.timeout = base_url, api_key, model, timeout

    def chat(self, messages: list, tools: list | None = None, temperature: float = 0.1) -> LLMResult:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"], payload["tool_choice"] = tools, "auto"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last = "no response"
        for attempt in range(4):
            t0 = time.perf_counter()
            try:
                r = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            except httpx.HTTPError as e:
                last = f"network error: {e}"
                time.sleep(2 ** attempt)
                continue
            latency = int((time.perf_counter() - t0) * 1000)
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(2 ** attempt)
                continue
            if r.status_code >= 400:
                raise LLMError(f"{self.model} returned HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            msg = data["choices"][0]["message"]
            usage = data.get("usage") or {}
            calls, raw = [], []
            for i, tc in enumerate(msg.get("tool_calls") or []):
                cid = tc.get("id") or f"call_{i}"
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append({"id": cid, "name": tc["function"]["name"], "arguments": args})
                raw.append({"id": cid, "type": "function", "function": {
                    "name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or "{}"}})
            return LLMResult(msg.get("content") or "", calls, raw, int(usage.get("prompt_tokens", 0)),
                             int(usage.get("completion_tokens", 0)), latency, self.model)
        raise LLMError(f"LLM request failed after retries: {last}")


class PatchContext:
    """Everything an engine needs; callables are wired up by the pipeline."""
    def __init__(self, **kw):
        self.state: dict = {}
        self.failure_log = ""
        self.__dict__.update(kw)


SYSTEM_PROMPT = """You are DevMind, an autonomous software engineer inside a sandboxed git workspace.
Resolve the issue by editing source files (do not edit tests unless the issue says the tests are wrong).
Workflow: localise with the repo map and `view`; reproduce with `run` (e.g. `python -m pytest -q`);
make the smallest correct edit with `str_replace` or `insert`; re-run the tests; call `finish` when they pass.
Edits to .py files are syntax-checked before being written: on AST_LINTER_ERROR fix the edit and retry.
One command per `run` call; no pipes, redirects or shell operators."""


class LLMEngine:
    uses_llm = True
    default_max_steps = 15

    def __init__(self, client: OpenAICompatClient, label: str):
        self.client, self.model, self.label = client, client.model, label

    def patch(self, ctx: PatchContext) -> bool:
        msgs = ctx.state.get("messages")
        log = (ctx.failure_log or "")[-6000:]
        if msgs is None:
            user = (f"## Issue\n{ctx.issue}\n\n## Repository map (AST)\n{render_repo_map_text(ctx.repomap)}\n\n"
                    f"## Failing test output\n{log or '(no failing output captured)'}")
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
            ctx.state["messages"] = msgs
        else:
            msgs.append({"role": "user", "content": f"The test suite still fails. Output:\n{log}\nFix what remains."})
        attempted = finished = False
        while not finished and not ctx.exhausted():
            res = self.client.chat(msgs, TOOL_SCHEMAS)
            ctx.record_llm(res)
            assistant: dict = {"role": "assistant", "content": res.content or None}
            if res.raw_tool_calls:
                assistant["tool_calls"] = res.raw_tool_calls
            msgs.append(assistant)
            if not res.tool_calls:
                break
            for call in res.tool_calls:
                attempted = True
                if call["name"] == "finish":
                    result, finished = "Acknowledged.", True
                else:
                    result = ctx.toolbox.dispatch(call["name"], call["arguments"])
                ctx.step(call["name"], call["arguments"], result)
                msgs.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                             "content": result[:8000]})
        return attempted


# ---------------------------------------------------------------- offline engine
OP_SWAPS = {">": [">=", "<"], "<": ["<=", ">"], ">=": [">", "<="], "<=": ["<", ">="], "==": ["!="],
            "!=": ["=="], "+": ["-"], "-": ["+"], "*": ["/"], "/": ["*"]}
NAME_SWAPS = {"and": ["or"], "or": ["and"], "min": ["max"], "max": ["min"], "True": ["False"], "False": ["True"]}


@dataclass
class Candidate:
    desc: str
    line: int
    new_source: str


def mutation_candidates(source: str, start: int, end: int) -> list:
    """Token-accurate single-edit mutations inside lines [start, end] (strings/comments untouched)."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    lines = source.splitlines(keepends=True)
    out = []
    for tok in toks:
        row, col = tok.start
        if not (start <= row <= end) or tok.end[0] != row:
            continue
        if tok.type == tokenize.OP:
            repl = OP_SWAPS.get(tok.string, [])
        elif tok.type == tokenize.NAME:
            repl = NAME_SWAPS.get(tok.string, [])
        elif tok.type == tokenize.NUMBER and tok.string.isdigit():
            n = int(tok.string)
            repl = [str(x) for x in dict.fromkeys([n - 1, n + 1, 0]) if x >= 0 and x != n]
        else:
            continue
        for r in repl:
            line = lines[row - 1]
            new_line = line[:col] + r + line[tok.end[1]:]
            new_src = "".join(lines[:row - 1]) + new_line + "".join(lines[row:])
            out.append(Candidate(f"`{tok.string}` -> `{r}` at line {row}", row, new_src))
    return out


def rank_suspects(repomap: dict, log: str, limit: int = 8) -> list:
    frames = [(f, int(n)) for f, n in re.findall(r"^([^\s:]+\.py):(\d+):", log or "", re.M)]
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", log or ""))
    scored = []
    for sym in repomap["symbols"]:
        if is_test_path(sym["file"]):
            continue
        name = sym["qualname"].split(".")[-1]
        score = sym["score"]
        if name in words and name != "__init__":
            score += 5
        for f, n in frames:
            if (sym["file"].endswith(f) or f.endswith(sym["file"])) and sym["lineno"] <= n <= sym["end_lineno"]:
                score += 10
        scored.append((score, sym))
    scored.sort(key=lambda x: -x[0])
    return [s for sc, s in scored if sc > 0][:limit] or [s for _, s in scored[:5]]


class DynamicSmartOfflineEngine:
    """No API key needed. Real execution-grounded search-based repair: localise suspects from the
    failing traceback + AST map, apply single-token mutations through the AST-gated editor, run the
    real test-suite for every candidate, keep the first patch that passes, undo the rest."""
    uses_llm = False
    default_max_steps = 60
    model = "none"
    label = "offline · mutation-repair"

    def patch(self, ctx: PatchContext) -> bool:
        tried = ctx.state.setdefault("tried", set())
        suspects = rank_suspects(ctx.repomap, ctx.failure_log)
        ctx.say("Suspects (traceback + AST relevance): " +
                ", ".join(f"{s['file']}::{s['qualname']}" for s in suspects[:5]) or "none")
        attempted = False
        for sym in suspects:
            src_path = ctx.toolbox.sb.safe_path(sym["file"])
            for cand in mutation_candidates(src_path.read_text(), sym["lineno"], sym["end_lineno"]):
                key = (sym["file"], cand.new_source)
                if key in tried:
                    continue
                if ctx.exhausted():
                    return attempted
                tried.add(key)
                attempted = True
                res = ctx.toolbox.write_file(sym["file"], cand.new_source)
                if res.startswith("AST_LINTER_ERROR"):
                    ctx.step("write_file", {"path": sym["file"], "edit": cand.desc}, res)
                    continue
                rc, out = ctx.run_tests(quiet=True)
                summary = next((l for l in reversed(out.strip().splitlines()) if l.strip()), "")
                verdict = "PASS" if rc == 0 else "fail"
                ctx.step("mutate+test", {"path": sym["file"], "symbol": sym["qualname"], "edit": cand.desc},
                         f"{verdict} - {summary}")
                if rc == 0:
                    ctx.say(f"Patch found: {cand.desc} in {sym['file']}::{sym['qualname']}")
                    return True
                ctx.toolbox.undo_edit(sym["file"])
        return attempted


def build_engine(config):
    if config.api_key:
        return LLMEngine(OpenAICompatClient(config.base_url, config.api_key, config.model), config.engine_label)
    return DynamicSmartOfflineEngine()
