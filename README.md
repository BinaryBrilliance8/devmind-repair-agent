# DevMind: LLM-based autonomous agent framework for automated software engineering

Pipeline: `LOCALIZING -> REPRODUCING -> PATCHING -> TESTING -> SELF_CORRECTING -> COMPLETE`.
Every phase change is emitted only after the real work for the previous phase finished.

## Run
    pip install -r requirements.txt
    python -m devmind                    # http://127.0.0.1:8000
    python -m devmind.sandbox.benchmarks # CLI benchmark, prints resolve rate
    python -m pytest tests -v

## Engines
* **Gemini / Groq** (OpenAI-compatible, tool calling). Set `GEMINI_API_KEY` or `GROQ_API_KEY`,
  or use *LLM settings* in the UI (`POST /api/config`). Real token usage and latency come from the API response.
* **DynamicSmartOfflineEngine** (no key): ranks suspect functions from the failing traceback + AST map,
  applies token-accurate single-edit mutations through the AST-gated editor, runs the real test-suite for each
  candidate and keeps the first patch that passes. This is search-based program repair, not an LLM.

## Components
* `agent/repomap.py` AST condenser + keyword relevance (`GET /api/sessions/{id}/repomap`)
* `agent/tools.py` view / str_replace / insert / undo_edit with a pre-write `ast.parse` gate and a git checkpoint before every write
* `agent/core.py` pipeline + sessions; `agent/llm.py` engines
* `sandbox/executor.py` allow-listed commands, `..`/absolute-path blocking, scrubbed env (`*API_KEY*`, `*SECRET*`, `*TOKEN*`...), rlimits, timeout, output truncation, real `git clone --depth 1`
* `sandbox/benchmarks.py` 4 SWE-bench-style templates (`calculator-pro`, `rate-limiter`, `jwt-auth`, `dag-scheduler`)
* Issue sources: template, GitHub URL (https only), local path (copied; original untouched)

## Honest limits
* The sandbox is process-level isolation. Run it inside a container/VM before pointing it at untrusted repositories.
* Templates are SWE-bench *style*. Real SWE-bench Lite instances need a dataset loader plus per-repo environments (next step).
* The Gemini/Groq path is tested against a mocked API (real clone, real edits, real tests); live calls need your key.
