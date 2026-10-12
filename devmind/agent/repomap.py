"""AST context condenser: symbol map + keyword relevance ranking for issue localisation."""
from __future__ import annotations

import ast
import re
from pathlib import Path

IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "env", ".tox", ".mypy_cache",
               ".pytest_cache", "build", "dist", ".idea", ".vscode", "site-packages"}
MAX_FILES, MAX_BYTES = 2000, 500_000
STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "not", "are", "but", "was", "when", "should",
             "returns", "return", "get", "set", "has", "have", "its", "into", "than", "then", "also", "too",
             "one", "all", "any", "can", "does", "bug", "issue", "error", "wrong", "fails", "instead"}


def split_words(text: str) -> list:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    return [w.lower() for w in re.split(r"[^A-Za-z0-9]+", text) if len(w) >= 3]


def keywords(query: str) -> set:
    return {w for w in split_words(query) if w not in STOPWORDS}


def is_test_path(rel: str) -> bool:
    p = rel.lower()
    name = p.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py" or "/tests/" in "/" + p


def _sig(node) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"


def _hits(text: str, kw: set) -> int:
    return len(set(split_words(text)) & kw)


def _analyse(source: str, rel: str, kw: set):
    tree = ast.parse(source)
    path_hits = _hits(rel, kw)
    imports, classes, functions, symbols = [], [], [], []
    score = path_hits * 3.0 + _hits(ast.get_docstring(tree) or "", kw)

    def make_sym(node, qual, kind):
        s = 2.0 * _hits(qual, kw) + _hits(ast.get_docstring(node) or "", kw) + 0.5 * path_hits
        return {"file": rel, "qualname": qual, "kind": kind, "signature": _sig(node),
                "lineno": node.lineno, "end_lineno": node.end_lineno, "score": round(s, 2)}

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
            score += 0.5 * _hits(ast.unparse(node), kw)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sym = make_sym(node, node.name, "function")
            symbols.append(sym)
            functions.append({"name": node.name, "signature": sym["signature"],
                              "lineno": sym["lineno"], "end_lineno": sym["end_lineno"]})
            score += 2.0 * _hits(node.name, kw) + _hits(ast.get_docstring(node) or "", kw)
        elif isinstance(node, ast.ClassDef):
            methods = []
            score += 2.0 * _hits(node.name, kw) + _hits(ast.get_docstring(node) or "", kw)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sym = make_sym(sub, f"{node.name}.{sub.name}", "method")
                    symbols.append(sym)
                    methods.append({"name": sub.name, "signature": sym["signature"],
                                    "lineno": sub.lineno, "end_lineno": sub.end_lineno})
                    score += 2.0 * _hits(sub.name, kw)
            classes.append({"name": node.name, "bases": [ast.unparse(b) for b in node.bases],
                            "lineno": node.lineno, "end_lineno": node.end_lineno, "methods": methods})
    test = is_test_path(rel)
    return {"path": rel, "is_test": test, "lines": source.count("\n") + 1,
            "score": round(score * (0.5 if test else 1.0), 2),
            "imports": imports, "classes": classes, "functions": functions}, symbols


def build_repo_map(workspace_path, issue_query: str = "") -> dict:
    root = Path(workspace_path).resolve()
    kw = keywords(issue_query)
    files, symbols, errors = [], [], []
    count = 0
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in IGNORE_DIRS for part in rel.parts):
            continue
        count += 1
        if count > MAX_FILES:
            break
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            info, syms = _analyse(path.read_text(errors="replace"), rel.as_posix(), kw)
        except SyntaxError as e:
            errors.append({"path": rel.as_posix(), "error": f"SyntaxError line {e.lineno}: {e.msg}"})
            continue
        files.append(info)
        symbols.extend(syms)
    files.sort(key=lambda f: (-f["score"], f["path"]))
    symbols.sort(key=lambda s: (-s["score"], s["file"], s["lineno"]))
    return {"workspace": str(root), "query_keywords": sorted(kw), "files": files, "symbols": symbols,
            "errors": errors, "stats": {"files": len(files), "symbols": len(symbols)}}


def render_repo_map_text(repomap: dict, max_files: int = 15, max_chars: int = 7000) -> str:
    out = []
    for f in repomap["files"][:max_files]:
        out.append(f"{f['path']}  (relevance {f['score']}, {f['lines']} lines)")
        for fn in f["functions"]:
            out.append(f"    L{fn['lineno']}-{fn['end_lineno']}  {fn['signature']}")
        for c in f["classes"]:
            out.append(f"    L{c['lineno']}-{c['end_lineno']}  class {c['name']}({', '.join(c['bases'])})")
            for m in c["methods"]:
                out.append(f"        L{m['lineno']}-{m['end_lineno']}  {m['signature']}")
    text = "\n".join(out)
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[map truncated]"
