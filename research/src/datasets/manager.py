"""
Dataset Manager for Legal NLP Research Framework.
Handles downloading, caching, and loading of legal datasets
from Hugging Face and custom local sources.
"""

import os
import json
import hashlib
import logging
import pickle
import ast
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import yaml

logger = logging.getLogger(__name__)

# pandas is imported lazily to avoid DLL load errors on systems with strict Application Control policies
pd = None
HAS_PANDAS = False

# ─────────────────────────────────────────────────────────────────────
# Known dataset registry
# ─────────────────────────────────────────────────────────────────────
DATASET_REGISTRY = {
    "case_hold": {
        "hf_name": "coastalcph/lex_glue",
        "hf_config": "case_hold",
        "task": "multi_class",
        "text_col": "context",
        "label_col": "label",
        "description": "CaseHOLD — Legal holding classification (LexGLUE)",
    },
    "ecthr_a": {
        "hf_name": "coastalcph/lex_glue",
        "hf_config": "ecthr_a",
        "task": "multi_label",
        "text_col": "text",
        "label_col": "labels",
        "description": "ECtHR-A — European Court of Human Rights articles (LexGLUE)",
    },
    "ecthr_b": {
        "hf_name": "coastalcph/lex_glue",
        "hf_config": "ecthr_b",
        "task": "multi_label",
        "text_col": "text",
        "label_col": "labels",
        "description": "ECtHR-B — ECtHR silver-rationale classification (LexGLUE)",
    },
    "eurlex": {
        "hf_name": "coastalcph/lex_glue",
        "hf_config": "eurlex",
        "task": "multi_label",
        "text_col": "text",
        "label_col": "labels",
        "description": "EURLEX — EU legislation classification (LexGLUE)",
    },
    "ledgar": {
        "hf_name": "coastalcph/lex_glue",
        "hf_config": "ledgar",
        "task": "multi_class",
        "text_col": "text",
        "label_col": "label",
        "description": "LEDGAR — Contract provision classification (LexGLUE)",
    },
    "scotus": {
        "hf_name": "coastalcph/lex_glue",
        "hf_config": "scotus",
        "task": "multi_class",
        "text_col": "text",
        "label_col": "label",
        "description": "SCOTUS — US Supreme Court decisions (LexGLUE)",
    },
    "unfair_tos": {
        "hf_name": "coastalcph/lex_glue",
        "hf_config": "unfair_tos",
        "task": "multi_label",
        "text_col": "text",
        "label_col": "labels",
        "description": "UNFAIR-ToS — Unfair clauses in Terms of Service (LexGLUE)",
    },
}


class DatasetManager:
    """
    Manages downloading, caching, and loading of legal NLP datasets.
    Supports HuggingFace datasets and custom local files.
    """

    def __init__(self, config: dict):
        self.config = config
        self.storage = config.get("storage", {})
        self.raw_dir = Path(self.storage.get("datasets_raw", "storage/datasets/raw"))
        self.processed_dir = Path(
            self.storage.get("datasets_processed", "storage/datasets/processed")
        )
        self.cache_dir = Path(
            self.storage.get("datasets_cache", "storage/datasets/cache")
        )
        self._setup_dirs()
        self._meta_file = self.cache_dir / "dataset_meta.json"
        self._meta = self._load_meta()

    def _setup_dirs(self):
        for d in [self.raw_dir, self.processed_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_meta(self) -> dict:
        if self._meta_file.exists():
            with open(self._meta_file) as f:
                return json.load(f)
        return {}

    def _save_meta(self):
        with open(self._meta_file, "w") as f:
            json.dump(self._meta, f, indent=2, default=str)

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def list_available(self) -> list[dict]:
        """Return list of all known datasets with download status."""
        result = []
        for key, info in DATASET_REGISTRY.items():
            entry = dict(info)
            entry["key"] = key
            entry["cached"] = key in self._meta
            entry["cache_time"] = self._meta.get(key, {}).get("cached_at", None)
            result.append(entry)
        return result

    def load(
        self,
        dataset_key: str,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        force_download: bool = False,
    ) -> dict:
        """
        Load a dataset by key. Downloads from HuggingFace if not cached.

        Returns:
            dict with keys: train, validation, test (each a list of dicts)
        """
        if dataset_key not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{dataset_key}'. "
                f"Available: {list(DATASET_REGISTRY.keys())}"
            )

        cache_path = self.cache_dir / f"{dataset_key}.pkl"

        if cache_path.exists() and not force_download:
            logger.info(f"Loading '{dataset_key}' from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
        else:
            logger.info(f"Downloading '{dataset_key}' from HuggingFace…")
            data = self._download_hf(dataset_key)
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            self._meta[dataset_key] = {
                "cached_at": datetime.now().isoformat(),
                "splits": list(data.keys()),
                "sizes": {k: len(v) for k, v in data.items()},
            }
            self._save_meta()
            logger.info(f"Cached '{dataset_key}' → {cache_path}")

        # Apply sample limits
        ds_cfg = self.config.get("datasets", {})
        if max_samples is None:
            limits = {
                "train": ds_cfg.get("max_train_samples"),
                "validation": ds_cfg.get("max_val_samples"),
                "test": ds_cfg.get("max_test_samples"),
            }
        else:
            limits = {"train": max_samples, "validation": max_samples, "test": max_samples}

        limited = {}
        for sp, rows in data.items():
            limit = limits.get(sp)
            limited[sp] = rows[:limit] if limit else rows

        if split:
            return {split: limited.get(split, [])}
        return limited

    def load_custom(
        self,
        file_path: str,
        text_col: Optional[str] = None,
        label_col: Optional[str] = None,
        task: str = "multi_class",
        dataset_name: Optional[str] = None,
    ) -> dict:
        """
        Load a custom dataset from CSV, JSON, JSONL, Excel, or Parquet.
        Auto-detects text/label columns if not specified.
        """
        # Lazy import pandas to avoid DLL load errors on systems with strict Application Control
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for custom dataset loading but is not available. "
                "Please install pandas or use HuggingFace datasets instead."
            ) from e

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        logger.info(f"Loading custom dataset from {path}")

        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in (".json",):
            df = pd.read_json(path)
        elif ext == ".jsonl":
            df = pd.read_json(path, lines=True)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        text_col = text_col or self._detect_text_col(df)
        label_col = label_col or self._detect_label_col(df)
        logger.info(f"Detected text_col='{text_col}', label_col='{label_col}'")

        rows = []
        for _, row in df.iterrows():
            label = row[label_col]
            if isinstance(label, str) and label.startswith("["):
                label = ast.literal_eval(label)
            rows.append({"text": str(row[text_col]), "label": label})

        # Simple 70/15/15 split
        n = len(rows)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        data = {
            "train": rows[:n_train],
            "validation": rows[n_train : n_train + n_val],
            "test": rows[n_train + n_val :],
        }

        # Register as custom dataset
        key = dataset_name or path.stem
        cache_path = self.cache_dir / f"custom_{key}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        self._meta[f"custom_{key}"] = {
            "source": str(path),
            "cached_at": datetime.now().isoformat(),
            "task": task,
            "text_col": text_col,
            "label_col": label_col,
            "splits": list(data.keys()),
            "sizes": {k: len(v) for k, v in data.items()},
        }
        self._save_meta()
        return data

    def delete_cache(self, dataset_key: str):
        """Delete cached version of a dataset."""
        cache_path = self.cache_dir / f"{dataset_key}.pkl"
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Deleted cache for '{dataset_key}'")
        if dataset_key in self._meta:
            del self._meta[dataset_key]
            self._save_meta()

    def get_dataset_info(self, dataset_key: str) -> dict:
        """Return metadata for a cached dataset."""
        return self._meta.get(dataset_key, {})

    def export_statistics(self, dataset_key: str, output_path: str):
        """Export dataset statistics to JSON."""
        from .statistics import DatasetStatistics

        data = self.load(dataset_key)
        info = DATASET_REGISTRY.get(dataset_key, {})
        stats = DatasetStatistics(data, info)
        stats_dict = stats.compute_all()
        with open(output_path, "w") as f:
            json.dump(stats_dict, f, indent=2, default=str)
        logger.info(f"Exported statistics → {output_path}")

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _download_hf(self, dataset_key: str) -> dict:
        """Download dataset from HuggingFace and convert to list-of-dicts."""
        from datasets import load_dataset as hf_load

        info = DATASET_REGISTRY[dataset_key]
        hf_name = info["hf_name"]
        hf_config = info.get("hf_config")
        text_col = info.get("text_col", "text")
        label_col = info.get("label_col", "label")

        if hf_config:
            hf_ds = hf_load(hf_name, hf_config, trust_remote_code=True)
        else:
            hf_ds = hf_load(hf_name, trust_remote_code=True)

        result = {}
        for split_name, split_data in hf_ds.items():
            rows = []
            for row in split_data:
                text = row.get(text_col, "")
                # Handle list-of-strings (some LexGLUE datasets)
                if isinstance(text, list):
                    text = " [SEP] ".join(str(t) for t in text)
                label = row.get(label_col, 0)
                rows.append({"text": str(text), "label": label})
            result[split_name] = rows
        return result

    def _detect_text_col(self, df) -> str:
        """Heuristic: find the most likely text column."""
        candidates = ["text", "content", "document", "body", "sentence", "context", "passage"]
        for c in candidates:
            if c in df.columns:
                return c
        # Fallback: column with longest average string length
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if str_cols:
            return max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
        return df.columns[0]

    def _detect_label_col(self, df) -> str:
        """Heuristic: find the most likely label column."""
        candidates = ["label", "labels", "category", "class", "target", "y"]
        for c in candidates:
            if c in df.columns:
                return c
        # Fallback: column with fewest unique values
        str_cols = [c for c in df.columns]
        return min(str_cols, key=lambda c: df[c].nunique())
