"""One configurable embedding model, used consistently for every chunk.

Backends
  sentence-transformers  local model (default BAAI/bge-small-en-v1.5)
  openai-compatible      any /v1/embeddings endpoint; credentials come from the
                         server environment (EMBEDDING_API_KEY, EMBEDDING_API_BASE)
                         and are never written to disk or sent to the browser.

Vectors are cached per (model, sha256(text)) in data/embeddings/<model-slug>/.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import yaml

from .fetch import ROOT

CFG = ROOT / "config" / "analysis.yaml"
EMB = ROOT / "data" / "embeddings"


def config() -> dict:
    return yaml.safe_load(CFG.read_text())


def preprocess(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t)).strip()


class Embedder:
    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or config())["embedding"]
        self.backend = self.cfg["backend"]
        self.model_name = self.cfg["model"]
        self.slug = re.sub(r"[^a-zA-Z0-9]+", "-", f"{self.backend}-{self.model_name}")
        self.dir = EMB / self.slug
        self.dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._cache_idx: dict[str, int] = {}
        self._cache_vecs: list[np.ndarray] = []
        self._load_cache()
        self.info = None

    # ---------------------------------------------------------------- cache
    def _load_cache(self):
        idx, vec = self.dir / "index.json", self.dir / "vectors.npy"
        if idx.exists() and vec.exists():
            self._cache_idx = json.loads(idx.read_text())
            arr = np.load(vec)
            self._cache_vecs = list(arr)

    def _save_cache(self):
        (self.dir / "index.json").write_text(json.dumps(self._cache_idx))
        np.save(self.dir / "vectors.npy", np.array(self._cache_vecs, dtype=np.float32))

    # ---------------------------------------------------------------- models
    def _st(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def model_info(self) -> dict:
        if self.info:
            return self.info
        info = dict(backend=self.backend, model=self.model_name, normalize=self.cfg["normalize"],
                    preprocessing=self.cfg["preprocessing"])
        if self.backend == "sentence-transformers":
            m = self._st()
            info["dimensions"] = m.get_sentence_embedding_dimension()
            info["max_seq_length"] = m.max_seq_length
            try:
                import sentence_transformers
                info["library"] = f"sentence-transformers {sentence_transformers.__version__}"
                from huggingface_hub import scan_cache_dir
                for repo in scan_cache_dir().repos:
                    if repo.repo_id == self.model_name:
                        info["revision"] = sorted(r.commit_hash for r in repo.revisions)[-1]
            except Exception as e:  # pragma: no cover
                info["revision_error"] = repr(e)
        else:
            info["api_base"] = os.environ.get("EMBEDDING_API_BASE", "https://api.openai.com/v1")
            info["dimensions"] = int(len(self._api(["dimension probe"])[0]))
        self.info = info
        return info

    def _api(self, texts: list[str]) -> np.ndarray:
        import requests
        key = os.environ.get("EMBEDDING_API_KEY")
        if not key:
            raise RuntimeError("EMBEDDING_API_KEY not set in server environment")
        base = os.environ.get("EMBEDDING_API_BASE", "https://api.openai.com/v1").rstrip("/")
        r = requests.post(f"{base}/embeddings", headers={"Authorization": f"Bearer {key}"},
                          json={"model": self.model_name, "input": texts}, timeout=120)
        r.raise_for_status()
        return np.array([d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])],
                        dtype=np.float32)

    # ---------------------------------------------------------------- public
    def embed(self, texts: list[str]) -> np.ndarray:
        texts = [preprocess(t) for t in texts]
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        todo = sorted({(k, t) for k, t in zip(keys, texts) if k not in self._cache_idx})
        if todo:
            bs = self.cfg.get("batch_size", 32)
            for i in range(0, len(todo), 256):
                part = todo[i:i + 256]
                if self.backend == "sentence-transformers":
                    vec = self._st().encode([t for _, t in part], batch_size=bs, normalize_embeddings=False,
                                            show_progress_bar=False, convert_to_numpy=True)
                else:
                    vec = np.vstack([self._api([t for _, t in part[j:j + bs]]) for j in range(0, len(part), bs)])
                for (k, _), v in zip(part, vec):
                    self._cache_idx[k] = len(self._cache_vecs)
                    self._cache_vecs.append(v.astype(np.float32))
            self._save_cache()
        out = np.array([self._cache_vecs[self._cache_idx[k]] for k in keys], dtype=np.float32)
        if self.cfg.get("normalize", True):
            out /= np.linalg.norm(out, axis=1, keepdims=True).clip(1e-12)
        return out
