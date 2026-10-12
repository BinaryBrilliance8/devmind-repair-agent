"""Runtime configuration: LLM provider, model, key. Keys live in memory only."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

WORKSPACE_ROOT = Path(os.environ.get("DEVMIND_WORKSPACES", "/tmp/devmind_workspaces"))

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "env": "GEMINI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "env": "GROQ_API_KEY",
    },
}


@dataclass
class Config:
    provider: str = "gemini"
    model: str = ""
    api_key: str = ""
    base_url: str = ""

    def __post_init__(self):
        self._fill()

    def _fill(self):
        spec = PROVIDERS.get(self.provider)
        if spec is None:
            raise ValueError(f"unknown provider '{self.provider}' (choose: {', '.join(PROVIDERS)})")
        self.model = self.model or spec["model"]
        self.base_url = self.base_url or spec["base_url"]

    @classmethod
    def from_env(cls) -> "Config":
        for name, spec in PROVIDERS.items():
            key = os.environ.get(spec["env"], "")
            if key:
                return cls(provider=name, api_key=key)
        return cls()

    def update(self, provider=None, model=None, api_key=None, base_url=None):
        if provider and provider != self.provider:
            if provider not in PROVIDERS:
                raise ValueError(f"unknown provider '{provider}'")
            self.provider, self.model, self.base_url = provider, "", ""
            if api_key is None:
                self.api_key = os.environ.get(PROVIDERS[provider]["env"], "")
        if model:
            self.model = model
        if api_key is not None:
            self.api_key = api_key.strip()
        if base_url:
            self.base_url = base_url
        self._fill()

    def snapshot(self) -> "Config":
        return Config(self.provider, self.model, self.api_key, self.base_url)

    @property
    def engine_label(self) -> str:
        return f"{self.provider} · {self.model}" if self.api_key else "offline · mutation-repair"

    def public(self) -> dict:
        return {"provider": self.provider, "model": self.model, "base_url": self.base_url,
                "has_key": bool(self.api_key), "engine": self.engine_label,
                "providers": {k: v["model"] for k, v in PROVIDERS.items()}}


CONFIG = Config.from_env()
