import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from devmind import config as cfg
from devmind.agent.core import SessionManager
from devmind.agent.llm import LLMResult, OpenAICompatClient, DynamicSmartOfflineEngine, mutation_candidates
from devmind.agent.repomap import build_repo_map
from devmind.agent.tools import Toolbox
from devmind.config import Config
from devmind.sandbox.benchmarks import TEMPLATES, materialize
from devmind.sandbox.executor import Sandbox, SandboxError

BUGGY_CALC = '''def average(nums):
    if not nums:
        raise ValueError("empty")
    return sum(nums) / (len(nums) + 1)
'''
CALC_TEST = '''import pytest
from calc import average

def test_avg():
    assert average([2, 4, 6]) == 4

def test_empty():
    with pytest.raises(ValueError):
        average([])
'''


@pytest.fixture(autouse=True)
def isolated_workspaces(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_ROOT", tmp_path / "workspaces")
    monkeypatch.setattr(cfg.CONFIG, "api_key", "")


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    materialize("calculator-pro", root)
    sb = Sandbox(root)
    sb.ensure_repo()
    return sb


def commit_count(sb):
    return int(sb.git("rev-list", "--count", "HEAD").output.strip())


# ---------------------------------------------------------------- AST gate
def test_ast_gate_rejects_syntax_error_and_leaves_disk_untouched(ws):
    tb = Toolbox(ws)
    before = (ws.root / "calc.py").read_text()
    commits = commit_count(ws)
    res = tb.str_replace("calc.py", "return total / (len(nums) + 1)", "return total / (len(nums) +")
    assert res.startswith("AST_LINTER_ERROR")
    assert (ws.root / "calc.py").read_text() == before
    assert commit_count(ws) == commits  # rejected edits don't even checkpoint


def test_valid_edit_is_checkpointed_then_written_and_undoable(ws):
    tb = Toolbox(ws)
    commits = commit_count(ws)
    assert tb.str_replace("calc.py", "(len(nums) + 1)", "len(nums)").startswith("OK")
    assert "total / len(nums)\n" in (ws.root / "calc.py").read_text()
    assert commit_count(ws) == commits + 1
    assert tb.undo_edit("calc.py").startswith("OK")
    assert "(len(nums) + 1)" in (ws.root / "calc.py").read_text()


def test_insert_tool(ws):
    tb = Toolbox(ws)
    assert tb.insert("calc.py", 0, "# header").startswith("OK")
    assert (ws.root / "calc.py").read_text().startswith("# header\n")


# ---------------------------------------------------------------- repomap
def test_repomap_extracts_symbols_and_ranks_relevant_file_first(tmp_path):
    (tmp_path / "billing.py").write_text('import os\n\nclass Invoice:\n    def total(self, items):\n        return 1\n\ndef helper():\n    pass\n')
    (tmp_path / "auth.py").write_text("def login(user):\n    return user\n")
    (tmp_path / "broken.py").write_text("def x(:\n")
    rm = build_repo_map(tmp_path, "Invoice total is wrong")
    assert rm["files"][0]["path"] == "billing.py"
    inv = rm["files"][0]["classes"][0]
    assert inv["name"] == "Invoice" and inv["lineno"] == 3
    assert inv["methods"][0]["name"] == "total" and inv["methods"][0]["lineno"] == 4
    assert any(s["qualname"] == "Invoice.total" for s in rm["symbols"])
    assert rm["errors"][0]["path"] == "broken.py"


# ---------------------------------------------------------------- sandbox
def test_sandbox_blocks_traversal_and_destructive(ws):
    for cmd in ["cat ../../etc/passwd", "rm -rf /", "cat /etc/passwd", "curl http://x", "python -c 'print(1)'",
                "ls ; ls", "git push", "find . -exec ls {} +"]:
        with pytest.raises(SandboxError):
            ws.run(cmd)
    with pytest.raises(SandboxError):
        ws.safe_path("../outside.txt")


def test_sandbox_scrubs_secrets(ws, monkeypatch):
    monkeypatch.setenv("MY_SECRET_API_KEY", "hunter2")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    (ws.root / "probe.py").write_text("import os\nprint(sorted(os.environ))\n")
    out = ws.run("python probe.py").output
    assert "MY_SECRET_API_KEY" not in out and "GITHUB_TOKEN" not in out and "PATH" in out


def test_sandbox_truncates_output_and_times_out(ws):
    (ws.root / "spam.py").write_text("print('x' * 500000)\n")
    res = ws.run("python spam.py")
    assert res.truncated and len(res.output) < 25_000
    (ws.root / "hang.py").write_text("import time\ntime.sleep(30)\n")
    res = ws.run("python hang.py", timeout=1)
    assert res.timed_out and res.returncode == 124


# ---------------------------------------------------------------- offline engine
def test_mutation_candidates_skip_strings_and_comments():
    src = 'def f(a):\n    s = "a > b"  # a > b\n    return a > 1\n'
    descs = [c.desc for c in mutation_candidates(src, 1, 3)]
    assert any("line 3" in d and "`>`" in d for d in descs)
    assert not any("line 2" in d and "`>`" in d for d in descs)


@pytest.mark.parametrize("tid", list(TEMPLATES))
def test_offline_engine_really_resolves_template(tid):
    mgr = SessionManager()
    t = TEMPLATES[tid]
    s = mgr.create(title=t["title"], description=t["issue"], template=tid, autostart=False)
    assert isinstance(s.engine, DynamicSmartOfflineEngine)
    s.run()
    assert s.status == "resolved", (s.error, [e for e in s.events if e["type"] in ("log", "result", "error")])
    # genuine work: real test output, ordered phases, non-trivial diff, fix on disk
    phases = [e["to"] for e in s.events if e["type"] == "phase_change"]
    assert phases[:4] == ["LOCALIZING", "REPRODUCING", "PATCHING", "TESTING"] and phases[-1] == "COMPLETE"
    term = "".join(e["text"] for e in s.events if e["type"] == "terminal_chunk")
    assert "failed" in term and "passed" in term
    assert "@@" in s.diff and s.telemetry["steps"] >= 1 and s.telemetry["tests_run"] >= 3
    check = Sandbox(s.workspace).run("python -m pytest -q -p no:cacheprovider")
    assert check.returncode == 0
    # tests were never edited
    assert "tests/" not in s.diff


# ---------------------------------------------------------------- real clone + mocked Gemini
def _tc(name, args, cid):
    return {"id": cid, "name": name, "arguments": args}


def _res(content="", calls=()):
    raw = [{"id": c["id"], "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]
    return LLMResult(content, list(calls), raw, prompt_tokens=100, completion_tokens=20, latency_ms=50,
                     model="gemini-2.5-flash")


def test_real_git_clone_edit_persists_with_mocked_gemini_api(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    (origin / "tests").mkdir(parents=True)
    (origin / "calc.py").write_text(BUGGY_CALC)
    (origin / "tests" / "test_calc.py").write_text(CALC_TEST)
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(origin)]
    for a in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run([*g, *a], check=True)
    monkeypatch.setenv("DEVMIND_ALLOW_FILE_URLS", "1")

    script = iter([
        _res("Let me read the file.", [_tc("view", {"path": "calc.py"}, "c1")]),
        _res("Trying a fix.", [_tc("str_replace", {"path": "calc.py", "old_str": "(len(nums) + 1)",
                                                   "new_str": "(len(nums) +"}, "c2")]),   # syntax error!
        _res("Syntax gate rejected it; retrying.", [_tc("str_replace", {"path": "calc.py", "old_str": "(len(nums) + 1)",
                                                                       "new_str": "len(nums)"}, "c3")]),
        _res("Done.", [_tc("finish", {"summary": "fixed divisor"}, "c4")]),
    ])
    seen = []
    monkeypatch.setattr(OpenAICompatClient, "chat",
                        lambda self, messages, tools=None, temperature=0.1: (seen.append(len(messages)), next(script))[1])

    mgr = SessionManager()
    s = mgr.create(title="average is wrong", description="average([2,4,6]) should be 4",
                   repo_url=f"file://{origin}", config=Config(provider="gemini", api_key="test-key"),
                   autostart=False)
    s.run()

    assert s.status == "resolved", (s.error, s.events[-5:])
    assert (s.workspace / ".git").exists()                                   # real clone happened
    assert "len(nums)\n" in (s.workspace / "calc.py").read_text() or "len(nums))" in (s.workspace / "calc.py").read_text()
    assert "+ 1" not in (s.workspace / "calc.py").read_text()                # edit physically on disk
    assert "+ 1" in (origin / "calc.py").read_text()                         # origin untouched
    tools = [e for e in s.events if e["type"] == "tool"]
    assert any(e["result"].startswith("AST_LINTER_ERROR") for e in tools)    # gate fired
    assert s.telemetry["total_tokens"] == 4 * 120 and s.telemetry["llm_latency_ms"] == 200
    assert seen == sorted(seen) and len(seen) == 4                           # conversation grows each turn
    log = Sandbox(s.workspace).git("log", "--oneline").output
    assert "checkpoint: before editing calc.py" in log                       # rollback checkpoint
    assert s.engine.label.startswith("gemini")


def test_llm_failure_surfaces_as_failed_not_fake_success(monkeypatch):
    def boom(self, *a, **k):
        from devmind.agent.llm import LLMError
        raise LLMError("HTTP 401 bad key")
    monkeypatch.setattr(OpenAICompatClient, "chat", boom)
    s = SessionManager().create(template="calculator-pro", config=Config(api_key="bad"), autostart=False)
    s.run()
    assert s.status == "failed" and "401" in s.error and s.resolved is False


# ---------------------------------------------------------------- HTTP API
def test_api_end_to_end(tmp_path):
    from devmind.main import app
    c = TestClient(app)
    assert c.get("/api/config").json()["has_key"] is False
    assert c.post("/api/issues", json={"template": "x"}).status_code == 400
    assert c.post("/api/issues", json={}).status_code == 400
    assert c.post("/api/issues", json={"repo_url": "http://evil/x"}).status_code == 400
    assert c.post("/api/issues", json={"workspace_path": "/definitely/not/here"}).status_code == 400
    sid = c.post("/api/issues", json={"template": "rate-limiter"}).json()["id"]
    for _ in range(300):
        st = c.get(f"/api/sessions/{sid}").json()
        if st["status"] in ("resolved", "unresolved", "failed"):
            break
        time.sleep(0.2)
    assert st["status"] == "resolved", st
    assert "@@" in st["diff"]
    rm = c.get(f"/api/sessions/{sid}/repomap").json()
    assert rm["files"][0]["path"] == "ratelimiter.py"
    names = [n["name"] for n in c.get(f"/api/sessions/{sid}/files").json()]
    assert "ratelimiter.py" in names
    assert c.get(f"/api/sessions/{sid}/file", params={"path": "ratelimiter.py"}).status_code == 200
    assert c.get(f"/api/sessions/{sid}/file", params={"path": "../../etc/passwd"}).status_code == 400
    with c.websocket_connect(f"/ws/sessions/{sid}") as sock:
        msg = sock.receive_json()
        assert msg["events"] and msg["state"]["status"] == "resolved"
    assert c.get("/").status_code == 200


def test_local_path_source_copies_and_never_touches_original(tmp_path):
    src = tmp_path / "proj"
    (src / "tests").mkdir(parents=True)
    (src / "calc.py").write_text(BUGGY_CALC)
    (src / "tests" / "test_calc.py").write_text(CALC_TEST)
    s = SessionManager().create(title="average wrong", description="average([2,4,6]) should be 4",
                                workspace_path=str(src), autostart=False)
    s.run()
    assert s.status == "resolved"
    assert "+ 1" in (src / "calc.py").read_text()
