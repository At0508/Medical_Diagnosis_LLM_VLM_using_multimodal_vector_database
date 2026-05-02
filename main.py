from __future__ import annotations

"""
Cluster-ready dermoscopy retrieval and classification pipeline with mandatory train-only augmentation, aggressive GPU support, and graceful local Llama/Ollama fallback.

Expected runtime layout:
.
├── main.py
├── prerequisites.py
├── requirements.txt
└── data/
    ├── bcn20000_metadata_2026-01-22.xlsx   # or .xls or .csv
    ├── ISIC-images/
    └── skin_lesion_rag_corpus_1000_lines.txt   # optional, can be .md/.csv/.xlsx/.xls

Default behavior:
- Loads metadata directly from ./data
- Matches metadata isic_id against image filename stems inside ./data/ISIC-images
- Splits matched rows into 92:8 train/test
- Builds image retrieval index on train images
- Optionally builds text chunk index from the local corpus for LLM-RAG only
- Runs selected branches: knn, llm_rag, vlm_rag
- Evaluates and writes predictions/metrics/plots under ./outputs
- Uses manifest-based invalidation so changed inputs do not silently reuse stale outputs
- Supports a self-test mode that uses lightweight local backends and a synthetic mini dataset

Production defaults:
- Image backend: CLIP
- Text backend: sentence-transformers
- LLM/VLM model: local Llama via Ollama

Offline/smoke-test options:
- --image-backend simple
- --text-backend simple
- --branches knn
- --self-test
"""

import argparse
import atexit
import math
import csv
import concurrent.futures
import gc
import hashlib
import importlib.util
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
import tempfile
import tarfile
import platform
import stat
import urllib.request
from urllib.parse import urlparse
import requests
from collections import Counter
import base64
from io import BytesIO
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    split_ratio_train: float = 0.80
    random_seed: int = 42
    stratify_target: str = "diagnosis_1"

    image_batch_size_cpu: int = 16
    image_batch_size_gpu: int = 96
    image_batch_size_t4: int = 64
    image_batch_size_a100: int = 192
    image_batch_size_h100: int = 256

    text_batch_size_cpu: int = 32
    text_batch_size_gpu: int = 192
    text_batch_size_t4: int = 128
    text_batch_size_a100: int = 512
    text_batch_size_h100: int = 768

    top_k_images: int = 25
    top_k_text_chunks: int = 4

    clip_model_name: str = "openai/clip-vit-base-patch32"
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_llm_model: str = "llama3.1:8b"
    ollama_vlm_model: str = "llama3.2-vision:11b"

    image_loader_workers: int = 4
    augmentation_workers: int = 8
    gpu_prefetch_factor: int = 2
    simple_image_size: int = 128
    faiss_enabled: bool = True
    aggressive_gpu_autotune: bool = True
    image_batch_probe_max: int = 256
    text_batch_probe_max: int = 1024

    augmentation_enabled: bool = True
    augmentation_copies_per_image: int = 1
    augmentation_output_subdir: str = "augmented_train_images"
    augmentation_max_rotation_deg: float = 15.0
    augmentation_crop_fraction: float = 0.06
    augmentation_brightness_jitter: float = 0.12
    augmentation_contrast_jitter: float = 0.12
    augmentation_color_jitter: float = 0.08
    augmentation_sharpness_jitter: float = 0.10
    augmentation_jpeg_quality: int = 95

    chunk_words: int = 180
    chunk_overlap_words: int = 40

    api_max_retries: int = 4
    api_retry_sleep: float = 3.0
    api_inter_request_sleep: float = 0.4

    ollama_start_timeout_seconds: int = 120
    ollama_healthcheck_interval_seconds: float = 1.5
    ollama_pull_timeout_seconds: int = 7200

    single_image_targets: Tuple[str, ...] = ("diagnosis_2",)


CONFIG = PipelineConfig()
TARGET_PRIORITY = [
    "diagnosis_1",
    "diagnosis_2",
    "diagnosis_3",
    "anatom_site_general",
    "sex",
    "melanocytic",
]
TEXT_COL_HINTS = {"text", "content", "article", "body", "note", "corpus", "passage"}
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Optional Hugging Face token for authenticated model downloads.
HARDCODED_HF_TOKEN = ""


def default_ollama_install_dir() -> Path:
    return Path(__file__).resolve().parent / ".local_ollama"


def default_ollama_models_dir() -> Path:
    return Path(__file__).resolve().parent / ".ollama_models"


def default_ollama_tmp_dir() -> Path:
    return Path(__file__).resolve().parent / ".ollama_tmp"


# -----------------------------------------------------------------------------
# CLI and logging
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster-ready dermoscopy RAG pipeline with local Llama/Ollama")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data", help="Input data directory")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs", help="Output directory")
    parser.add_argument(
        "--branches",
        nargs="+",
        default=["knn", "llm_rag", "vlm_rag"],
        choices=["knn", "llm_rag", "vlm_rag"],
        help="Branches to run",
    )
    parser.add_argument(
        "--image-backend",
        default="clip",
        choices=["clip", "simple"],
        help="Image embedding backend. 'simple' is useful for smoke tests or offline runs.",
    )
    parser.add_argument(
        "--text-backend",
        default="sentence_transformer",
        choices=["sentence_transformer", "simple"],
        help="Text embedding backend. 'simple' uses TF-IDF.",
    )
    parser.add_argument("--disable-local-llm", action="store_true", help="Skip Ollama/local-LLM dependent branches")
    parser.add_argument("--ollama-base-url", type=str, default=CONFIG.ollama_base_url, help="Ollama API base URL")
    parser.add_argument("--ollama-llm-model", type=str, default=CONFIG.ollama_llm_model, help="Ollama text model name")
    parser.add_argument("--ollama-vlm-model", type=str, default=CONFIG.ollama_vlm_model, help="Ollama vision model name")
    parser.add_argument("--ollama-install-dir", type=Path, default=default_ollama_install_dir(), help="Local directory where Ollama should be installed if not already on PATH")
    parser.add_argument("--ollama-models-dir", type=Path, default=default_ollama_models_dir(), help="Directory for local Ollama models")
    parser.add_argument("--ollama-tmp-dir", type=Path, default=default_ollama_tmp_dir(), help="Writable temporary directory for Ollama runtime")
    parser.add_argument("--ollama-install-url", type=str, default="", help="Optional explicit Ollama tarball URL for local installation")
    parser.add_argument("--ollama-keep-alive", type=str, default="30m", help="keep_alive value sent to Ollama chat API")
    parser.add_argument("--allow-ollama-fallback", action=argparse.BooleanOptionalAction, default=False, help="If Ollama is unavailable, continue with the available non-Ollama branches instead of failing. Default is strict: fail instead of silently degrading.")
    parser.add_argument("--require-ollama", action="store_true", help="Fail fast if requested Ollama branches or models are unavailable.")
    parser.add_argument("--fail-on-missing-local-branches", action=argparse.BooleanOptionalAction, default=True, help="If llm_rag or vlm_rag was requested but could not be activated and warmed up, raise an error instead of silently dropping the branch.")
    parser.add_argument("--auto-install-ollama", action=argparse.BooleanOptionalAction, default=True, help="Install a local Ollama runtime under the project directory when no ollama binary is available.")
    parser.add_argument("--auto-start-ollama", action=argparse.BooleanOptionalAction, default=True, help="Start a local Ollama server from main.py when the base URL is local and no server is reachable.")
    parser.add_argument("--auto-pull-ollama-models", action=argparse.BooleanOptionalAction, default=True, help="If required local Ollama models are missing, attempt to pull them before running the affected branches.")
    parser.add_argument("--skip-single-image-demo", action="store_true", help="Skip single-image demo prediction")
    parser.add_argument("--self-test", action="store_true", help="Create and run a synthetic smoke test")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


LOGGER = logging.getLogger("dermoscopy_rag")


# -----------------------------------------------------------------------------
# Optional dependency helpers
# -----------------------------------------------------------------------------


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def lazy_import_torch():
    if not module_available("torch"):
        raise RuntimeError("torch is not installed. Install it from requirements.txt.")
    import torch

    return torch


def lazy_import_clip():
    if not module_available("transformers"):
        raise RuntimeError("transformers is not installed. Install it from requirements.txt.")
    from transformers import CLIPModel, CLIPProcessor

    return CLIPModel, CLIPProcessor


def lazy_import_sentence_transformer():
    if not module_available("sentence_transformers"):
        raise RuntimeError("sentence-transformers is not installed. Install it from requirements.txt.")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def lazy_import_faiss():
    try:
        import faiss
        return faiss
    except Exception as exc:
        raise RuntimeError("faiss is not installed. Install faiss-cpu from requirements.txt.") from exc




def configure_optional_hf_token() -> None:
    token = (HARDCODED_HF_TOKEN or "").strip()
    if not token:
        token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)


# -----------------------------------------------------------------------------
# Paths and filesystem helpers
# -----------------------------------------------------------------------------


@dataclass
class RuntimePaths:
    data_dir: Path
    output_dir: Path
    artifacts_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    plots_dir: Path
    cache_dir: Path
    manifest_path: Path


def build_paths(data_dir: Path, output_dir: Path) -> RuntimePaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    predictions_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    cache_dir = output_dir / "cache"
    for d in [artifacts_dir, predictions_dir, metrics_dir, plots_dir, cache_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(
        data_dir=data_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        predictions_dir=predictions_dir,
        metrics_dir=metrics_dir,
        plots_dir=plots_dir,
        cache_dir=cache_dir,
        manifest_path=artifacts_dir / "run_manifest.json",
    )


def require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def find_metadata_file(data_dir: Path) -> Path:
    candidates: List[Path] = []
    for ext in (".xlsx", ".xls", ".csv"):
        candidates.extend(data_dir.glob(f"bcn20000_metadata_2026-01-22{ext}"))
    if not candidates:
        for p in data_dir.iterdir():
            if p.is_file() and p.stem == "bcn20000_metadata_2026-01-22":
                candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            "Could not find metadata file in ./data with stem 'bcn20000_metadata_2026-01-22'"
        )
    return sorted(candidates)[0]


def find_text_file(data_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for ext in (".txt", ".md", ".csv", ".xlsx", ".xls"):
        candidates.extend(data_dir.glob(f"skin_lesion_rag_corpus_1000_lines{ext}"))
    return sorted(candidates)[0] if candidates else None


def image_root(data_dir: Path) -> Path:
    root = data_dir / "ISIC-images"
    require_exists(root, "image folder")
    return root


# -----------------------------------------------------------------------------
# Runtime helpers
# -----------------------------------------------------------------------------


def runtime_info() -> Dict[str, Any]:
    info = {
        "cuda_available": False,
        "device": "cpu",
        "gpu_name": None,
        "gpu_total_mem_gb": 0.0,
        "is_t4": False,
        "is_a100": False,
        "is_h100": False,
        "amp_dtype": "float32",
    }
    if module_available("torch"):
        torch = lazy_import_torch()
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["device"] = "cuda" if info["cuda_available"] else "cpu"
        if info["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            gpu_name = torch.cuda.get_device_name(0)
            lower_name = gpu_name.lower()
            info["gpu_name"] = gpu_name
            info["gpu_total_mem_gb"] = round(props.total_memory / (1024 ** 3), 2)
            info["is_t4"] = "t4" in lower_name
            info["is_a100"] = "a100" in lower_name
            info["is_h100"] = "h100" in lower_name
            info["amp_dtype"] = "bfloat16" if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else "float16"
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
    return info


def _env_int(name: str) -> Optional[int]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception as exc:
        raise ValueError(f"Environment override {name} must be an integer, got: {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment override {name} must be positive, got: {value}")
    return value


def available_cpu_count() -> int:
    candidates: List[int] = []
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except Exception:
        pass
    for env_name in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        raw = (os.getenv(env_name) or "").strip()
        if raw.isdigit():
            candidates.append(int(raw))
    cpu_total = os.cpu_count() or 1
    candidates.append(cpu_total)
    valid = [c for c in candidates if c and c > 0]
    return max(1, min(valid) if valid else cpu_total)


def safe_image_loader_workers(device: str, batch_size: int) -> int:
    override = os.getenv("DERM_IMAGE_LOADER_WORKERS")
    if override is not None and override.strip() == "0":
        return 0
    if device != "cuda":
        return 0
    available = available_cpu_count()
    requested = _env_int("DERM_IMAGE_LOADER_WORKERS") or CONFIG.image_loader_workers
    hard_cap = 4
    if batch_size >= 256:
        hard_cap = 1
    elif batch_size >= 128:
        hard_cap = 2
    workers = min(requested, hard_cap, max(0, available - 1))
    return max(0, workers)


def safe_image_batch_size(runtime: Dict[str, Any], requested: int) -> int:
    override = _env_int("DERM_IMAGE_BATCH_SIZE")
    if override is not None:
        return override
    if not runtime.get("cuda_available"):
        return min(requested, CONFIG.image_batch_size_cpu)
    if runtime.get("is_h100"):
        cap = CONFIG.image_batch_size_h100
    elif runtime.get("is_a100") or runtime.get("gpu_total_mem_gb", 0.0) >= 39.0:
        cap = CONFIG.image_batch_size_a100
    elif runtime.get("is_t4"):
        cap = CONFIG.image_batch_size_t4
    else:
        cap = CONFIG.image_batch_size_gpu
    return max(1, min(requested, cap, CONFIG.image_batch_probe_max))


def safe_text_batch_size(runtime: Dict[str, Any], requested: int) -> int:
    override = _env_int("DERM_TEXT_BATCH_SIZE")
    if override is not None:
        return override
    if not runtime.get("cuda_available"):
        return min(requested, CONFIG.text_batch_size_cpu)
    if runtime.get("is_h100"):
        cap = CONFIG.text_batch_size_h100
    elif runtime.get("is_a100") or runtime.get("gpu_total_mem_gb", 0.0) >= 39.0:
        cap = CONFIG.text_batch_size_a100
    elif runtime.get("is_t4"):
        cap = CONFIG.text_batch_size_t4
    else:
        cap = CONFIG.text_batch_size_gpu
    return max(1, min(requested, cap, CONFIG.text_batch_probe_max))


def tuned_image_batch_size(rt: Dict[str, Any]) -> int:
    if not rt["cuda_available"]:
        return CONFIG.image_batch_size_cpu
    requested = CONFIG.image_batch_size_h100 if rt.get("is_h100") else CONFIG.image_batch_size_a100 if (rt.get("is_a100") or rt.get("gpu_total_mem_gb", 0.0) >= 39.0) else CONFIG.image_batch_size_t4 if rt["is_t4"] else CONFIG.image_batch_size_gpu
    return safe_image_batch_size(rt, requested)


def tuned_text_batch_size(rt: Dict[str, Any]) -> int:
    if not rt["cuda_available"]:
        return CONFIG.text_batch_size_cpu
    requested = CONFIG.text_batch_size_h100 if rt.get("is_h100") else CONFIG.text_batch_size_a100 if (rt.get("is_a100") or rt.get("gpu_total_mem_gb", 0.0) >= 39.0) else CONFIG.text_batch_size_t4 if rt["is_t4"] else CONFIG.text_batch_size_gpu
    return safe_text_batch_size(rt, requested)


def print_runtime_summary(rt: Dict[str, Any]) -> None:
    LOGGER.info("=" * 80)
    LOGGER.info("Runtime summary")
    LOGGER.info("=" * 80)
    LOGGER.info("Device: %s", rt["device"])
    LOGGER.info("CUDA available: %s", rt["cuda_available"])
    LOGGER.info("GPU name: %s", rt["gpu_name"])
    LOGGER.info("GPU total memory (GB): %s", rt.get("gpu_total_mem_gb"))
    LOGGER.info("T4 optimized: %s", rt["is_t4"])
    LOGGER.info("A100 optimized: %s", rt.get("is_a100"))
    LOGGER.info("H100 optimized: %s", rt.get("is_h100"))
    LOGGER.info("Autocast dtype: %s", rt.get("amp_dtype"))
    LOGGER.info("Image batch size: %s", tuned_image_batch_size(rt))
    LOGGER.info("Text batch size: %s", tuned_text_batch_size(rt))
    if rt["cuda_available"] and module_available("torch"):
        torch = lazy_import_torch()
        LOGGER.info("CUDA memory allocated: %.2f GB", torch.cuda.memory_allocated() / 1e9)
        LOGGER.info("CUDA memory reserved:  %.2f GB", torch.cuda.memory_reserved() / 1e9)
    LOGGER.info("=" * 80)


def clean_cuda() -> None:
    gc.collect()
    if module_available("torch"):
        torch = lazy_import_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_pickle(obj: Any, path: Path) -> None:
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Manifest invalidation
# -----------------------------------------------------------------------------


def build_run_manifest(
    metadata_path: Path,
    images_dir: Path,
    text_path: Optional[Path],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    image_files = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
    )
    img_meta = [(str(p.relative_to(images_dir)), p.stat().st_size) for p in image_files]
    return {
        "config": asdict(CONFIG),
        "args": {
            "branches": sorted(args.branches),
            "image_backend": args.image_backend,
            "text_backend": args.text_backend,
            "disable_local_llm": args.disable_local_llm,
            "ollama_base_url": args.ollama_base_url,
            "ollama_llm_model": args.ollama_llm_model,
            "ollama_vlm_model": args.ollama_vlm_model,
            "self_test": args.self_test,
        },
        "metadata_file": str(metadata_path.name),
        "metadata_sha256": sha256_of_file(metadata_path),
        "images_dir": str(images_dir.name),
        "image_count": len(image_files),
        "image_listing_sha256": hashlib.sha256(
            json.dumps(img_meta, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "text_file": str(text_path.name) if text_path else None,
        "text_sha256": sha256_of_file(text_path) if text_path else None,
    }


def clear_outputs_for_manifest_change(paths: RuntimePaths) -> None:
    LOGGER.info("Input manifest changed. Clearing stale outputs, cache, and artifacts.")
    for d in [paths.predictions_dir, paths.metrics_dir, paths.plots_dir, paths.cache_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for p in paths.artifacts_dir.iterdir():
        if p.name == paths.manifest_path.name:
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def maybe_invalidate_on_manifest_change(
    metadata_path: Path,
    images_dir: Path,
    text_path: Optional[Path],
    args: argparse.Namespace,
    paths: RuntimePaths,
) -> None:
    new_manifest = build_run_manifest(metadata_path, images_dir, text_path, args)
    if paths.manifest_path.exists():
        old_manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
        if old_manifest != new_manifest:
            clear_outputs_for_manifest_change(paths)
    paths.manifest_path.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def normalize_text(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def normalize_boolish(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    truthy = {"1", "true", "yes", "y", "t", "melanocytic", "melanocyte", "positive"}
    falsy = {"0", "false", "no", "n", "f", "non-melanocytic", "negative"}
    if s in truthy:
        return "true"
    if s in falsy:
        return "false"
    return str(x).strip()


def prepare_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns}).copy()

    id_col = None
    for col in df.columns:
        if col.lower() == "isic_id":
            id_col = col
            break
    if id_col is None:
        raise ValueError("Metadata must contain an 'isic_id' column.")
    if id_col != "isic_id":
        df = df.rename(columns={id_col: "isic_id"})

    df["isic_id"] = df["isic_id"].astype(str).str.strip()
    for col in df.columns:
        if col == "melanocytic":
            df[col] = df[col].map(normalize_boolish)
        else:
            df[col] = df[col].map(normalize_text)
    return df.drop_duplicates(subset=["isic_id"]).reset_index(drop=True)


def collect_image_paths(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS])


def build_image_map(paths: Sequence[Path]) -> Tuple[Dict[str, str], List[str]]:
    image_map: Dict[str, str] = {}
    duplicates: List[str] = []
    for path in paths:
        stem = path.stem.strip()
        if stem in image_map:
            duplicates.append(stem)
            continue
        image_map[stem] = str(path)
    return image_map, sorted(set(duplicates))


def attach_image_paths(metadata: pd.DataFrame, image_map: Dict[str, str]) -> pd.DataFrame:
    merged = metadata.copy()
    merged["image_path"] = merged["isic_id"].map(image_map)
    return merged


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stratify = None
    col = CONFIG.stratify_target
    if col in df.columns:
        vc = df[col].fillna("__MISSING__").value_counts()
        if len(vc) > 1 and vc.min() >= 2:
            stratify = df[col].fillna("__MISSING__")
    try:
        train_df, test_df = train_test_split(
            df,
            train_size=CONFIG.split_ratio_train,
            random_state=CONFIG.random_seed,
            shuffle=True,
            stratify=stratify,
        )
    except Exception:
        train_df, test_df = train_test_split(
            df,
            train_size=CONFIG.split_ratio_train,
            random_state=CONFIG.random_seed,
            shuffle=True,
            stratify=None,
        )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Mandatory train-only offline augmentation
# -----------------------------------------------------------------------------


def _stable_int_seed(*parts: Any) -> int:
    payload = "||".join(str(x) for x in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**32)


def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def augmentation_config_dict() -> Dict[str, Any]:
    return {
        "enabled": CONFIG.augmentation_enabled,
        "copies_per_image": CONFIG.augmentation_copies_per_image,
        "output_subdir": CONFIG.augmentation_output_subdir,
        "max_rotation_deg": CONFIG.augmentation_max_rotation_deg,
        "crop_fraction": CONFIG.augmentation_crop_fraction,
        "brightness_jitter": CONFIG.augmentation_brightness_jitter,
        "contrast_jitter": CONFIG.augmentation_contrast_jitter,
        "color_jitter": CONFIG.augmentation_color_jitter,
        "sharpness_jitter": CONFIG.augmentation_sharpness_jitter,
        "jpeg_quality": CONFIG.augmentation_jpeg_quality,
    }


def augment_pil_image(img: Image.Image, rng: np.random.Generator, cfg: Dict[str, Any]) -> Image.Image:
    img = _ensure_rgb(img)
    if float(rng.random()) < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if float(rng.random()) < 0.15:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    angle = float(rng.uniform(-cfg["max_rotation_deg"], cfg["max_rotation_deg"]))
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR)
    w, h = img.size
    crop_frac = float(rng.uniform(0.0, cfg["crop_fraction"]))
    dx = int(round(w * crop_frac))
    dy = int(round(h * crop_frac))
    if dx > 0 and dy > 0 and (w - 2 * dx) >= 16 and (h - 2 * dy) >= 16:
        left = int(rng.integers(0, dx + 1))
        top = int(rng.integers(0, dy + 1))
        right = w - int(rng.integers(0, dx + 1))
        bottom = h - int(rng.integers(0, dy + 1))
        if right - left >= 16 and bottom - top >= 16:
            img = img.crop((left, top, right, bottom)).resize((w, h), Image.Resampling.BILINEAR)
    from PIL import ImageEnhance
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(1.0 - cfg["brightness_jitter"], 1.0 + cfg["brightness_jitter"])))
    img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(1.0 - cfg["contrast_jitter"], 1.0 + cfg["contrast_jitter"])))
    img = ImageEnhance.Color(img).enhance(float(rng.uniform(1.0 - cfg["color_jitter"], 1.0 + cfg["color_jitter"])))
    img = ImageEnhance.Sharpness(img).enhance(float(rng.uniform(1.0 - cfg["sharpness_jitter"], 1.0 + cfg["sharpness_jitter"])))
    return img


def create_augmented_train_df(train_df: pd.DataFrame, output_dir: Path, copies_per_image: int, cfg: Dict[str, Any]) -> pd.DataFrame:
    if copies_per_image < 1:
        raise ValueError("copies_per_image must be >= 1")
    if "image_path" not in train_df.columns:
        raise ValueError("train_df must contain image_path")

    output_dir.mkdir(parents=True, exist_ok=True)

    def _process_row(row_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        image_path = Path(str(row_dict["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for augmentation: {image_path}")
        out_rows = []
        with Image.open(image_path) as src_img:
            src_img = _ensure_rgb(src_img)
            for copy_idx in range(1, copies_per_image + 1):
                seed = _stable_int_seed(CONFIG.random_seed, row_dict.get("isic_id"), copy_idx, image_path.name)
                rng = np.random.default_rng(seed)
                aug_img = augment_pil_image(src_img.copy(), rng, cfg)
                out_path = output_dir / f"{image_path.stem}__aug{copy_idx}.jpg"
                aug_img.save(out_path, format="JPEG", quality=int(cfg["jpeg_quality"]), optimize=True)
                aug_row = dict(row_dict)
                aug_row["source_isic_id"] = row_dict.get("isic_id")
                aug_row["isic_id"] = f"{row_dict.get('isic_id')}__aug{copy_idx}"
                aug_row["image_path"] = str(out_path)
                aug_row["is_augmented"] = True
                aug_row["augmentation_copy_idx"] = copy_idx
                aug_row["augmentation_seed"] = int(seed)
                out_rows.append(aug_row)
        return out_rows

    base_df = train_df.copy()
    base_df["is_augmented"] = False
    base_df["source_isic_id"] = base_df["isic_id"]
    base_df["augmentation_copy_idx"] = 0
    base_df["augmentation_seed"] = pd.NA

    row_dicts = [row.to_dict() for _, row in train_df.iterrows()]
    workers = min(CONFIG.augmentation_workers, max(1, os.cpu_count() or 1))
    augmented_rows: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_row, row_dict) for row_dict in row_dicts]
        for fut in concurrent.futures.as_completed(futures):
            augmented_rows.extend(fut.result())

    aug_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([base_df, aug_df], ignore_index=True) if not aug_df.empty else base_df
    return combined.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Text corpus helpers
# -----------------------------------------------------------------------------


def load_text_sources(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["source_id", "text"])

    suffix = path.suffix.lower()
    rows: List[Dict[str, str]] = []

    if suffix in {".txt", ".md"}:
        rows.append({"source_id": path.stem, "text": path.read_text(encoding="utf-8", errors="ignore")})
    elif suffix == ".csv":
        df = pd.read_csv(path)
        text_cols = [c for c in df.columns if c.strip().lower() in TEXT_COL_HINTS] or [df.columns[0]]
        for i, row in df.iterrows():
            texts = [str(row[c]) for c in text_cols if pd.notna(row[c]) and str(row[c]).strip()]
            if texts:
                rows.append({"source_id": f"{path.stem}_{i}", "text": "\n".join(texts)})
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
        text_cols = [c for c in df.columns if c.strip().lower() in TEXT_COL_HINTS] or [df.columns[0]]
        for i, row in df.iterrows():
            texts = [str(row[c]) for c in text_cols if pd.notna(row[c]) and str(row[c]).strip()]
            if texts:
                rows.append({"source_id": f"{path.stem}_{i}", "text": "\n".join(texts)})
    else:
        raise ValueError(f"Unsupported text file format: {path}")

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["source_id", "text"])
    out["text"] = out["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return out[out["text"].str.len() > 0].reset_index(drop=True)


def chunk_text_sources(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    if df.empty:
        return pd.DataFrame(columns=["chunk_id", "source_id", "chunk_text"])

    step = max(1, CONFIG.chunk_words - CONFIG.chunk_overlap_words)
    for _, row in df.iterrows():
        words = str(row["text"]).split()
        for start in range(0, len(words), step):
            chunk = words[start : start + CONFIG.chunk_words]
            if chunk:
                rows.append(
                    {
                        "chunk_id": f"{row['source_id']}__{start}",
                        "source_id": row["source_id"],
                        "chunk_text": " ".join(chunk),
                    }
                )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Target and label vocab helpers
# -----------------------------------------------------------------------------


def infer_active_targets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
    target = "diagnosis_2"
    if target in train_df.columns and target in test_df.columns:
        if train_df[target].notna().any() and test_df[target].notna().any():
            return [target]
    return []


def build_label_vocab_train_only(train_df: pd.DataFrame, targets: Sequence[str]) -> Dict[str, List[str]]:
    vocab: Dict[str, List[str]] = {}
    for target in targets:
        vocab[target] = sorted(
            set(str(x).strip() for x in train_df[target].dropna().tolist() if str(x).strip())
        )
    return vocab


# -----------------------------------------------------------------------------
# Generic similarity index
# -----------------------------------------------------------------------------


class SimilarityIndex:
    def __init__(self) -> None:
        self.vectors: Optional[np.ndarray] = None
        self.meta: Optional[pd.DataFrame] = None
        self.faiss_index: Optional[Any] = None
        self.use_faiss: bool = False

    def build(self, vectors: np.ndarray, meta: pd.DataFrame) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        self.vectors = np.asarray(vectors, dtype="float32")
        self.meta = meta.reset_index(drop=True).copy()
        self.faiss_index = None
        self.use_faiss = False

        if CONFIG.faiss_enabled and module_available("faiss"):
            try:
                faiss = lazy_import_faiss()
                index = faiss.IndexFlatIP(self.vectors.shape[1])
                index.add(self.vectors)
                self.faiss_index = index
                self.use_faiss = True
            except Exception as exc:
                LOGGER.warning("FAISS unavailable or failed to initialize (%s). Falling back to numpy search.", exc)
                self.faiss_index = None
                self.use_faiss = False

    def _rows_from_search(self, sims: np.ndarray, idxs: np.ndarray) -> pd.DataFrame:
        rows = []
        for idx, sim in zip(idxs.tolist(), sims.tolist()):
            row = self.meta.iloc[int(idx)].to_dict()
            row["similarity"] = float(sim)
            rows.append(row)
        return pd.DataFrame(rows)

    def search_by_vector(self, query_vec: np.ndarray, top_k: int) -> pd.DataFrame:
        if self.vectors is None or self.meta is None:
            raise RuntimeError("Similarity index has not been built.")
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        if self.use_faiss and self.faiss_index is not None:
            sims, idxs = self.faiss_index.search(q, top_k)
            sims = sims[0]
            idxs = idxs[0]
            valid = idxs >= 0
            return self._rows_from_search(sims[valid], idxs[valid])
        sims = (q @ self.vectors.T).reshape(-1)
        idxs = np.argsort(-sims)[:top_k]
        return self._rows_from_search(sims[idxs], idxs)

    def search_many_by_vectors(self, query_vecs: np.ndarray, top_k: int) -> List[pd.DataFrame]:
        if self.vectors is None or self.meta is None:
            raise RuntimeError("Similarity index has not been built.")
        q = np.asarray(query_vecs, dtype="float32")
        out: List[pd.DataFrame] = []
        if self.use_faiss and self.faiss_index is not None:
            sims, idxs = self.faiss_index.search(q, top_k)
            for i in range(q.shape[0]):
                valid = idxs[i] >= 0
                out.append(self._rows_from_search(sims[i][valid], idxs[i][valid]))
            return out
        sim_mat = q @ self.vectors.T
        top = np.argsort(-sim_mat, axis=1)[:, :top_k]
        for i in range(q.shape[0]):
            idxs = top[i]
            out.append(self._rows_from_search(sim_mat[i, idxs], idxs))
        return out


# -----------------------------------------------------------------------------
# Embedding backends
# -----------------------------------------------------------------------------


class ClipImageEmbedder:
    def __init__(self, model_name: str, runtime: Dict[str, Any]) -> None:
        configure_optional_hf_token()
        torch = lazy_import_torch()
        CLIPModel, CLIPProcessor = lazy_import_clip()
        self.torch = torch
        self.device = runtime["device"]
        self.runtime = runtime
        self.amp_dtype = torch.bfloat16 if runtime.get("amp_dtype") == "bfloat16" else torch.float16
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._autotuned_batch_size: Optional[int] = None
        if self.device == "cuda":
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception:
                pass

    def _unwrap_clip_output(self, raw: Any) -> Any:
        if hasattr(raw, "float"):
            return raw
        for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
            value = getattr(raw, attr, None)
            if value is not None:
                return value
        if isinstance(raw, (tuple, list)) and len(raw) > 0:
            return raw[0]
        raise RuntimeError(f"Unexpected CLIP output type: {type(raw)!r}")

    def _encode_image_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self.torch
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    raw = self.model.get_image_features(**inputs)
            else:
                raw = self.model.get_image_features(**inputs)
        feats = self._unwrap_clip_output(raw)
        if hasattr(feats, "ndim") and feats.ndim > 2:
            feats = feats[:, 0, :]
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.detach().cpu().numpy().astype("float32")

    def _autotune_batch_size(self, paths: Sequence[str], initial_batch_size: int) -> int:
        requested = safe_image_batch_size(self.runtime, initial_batch_size)
        if self.device != "cuda" or not CONFIG.aggressive_gpu_autotune or self._autotuned_batch_size is not None or not paths:
            return requested if self._autotuned_batch_size is None else self._autotuned_batch_size
        torch = self.torch
        max_probe = safe_image_batch_size(self.runtime, max(requested, CONFIG.image_batch_probe_max))
        probe_path = str(paths[0])
        sample = Image.open(probe_path).convert("RGB")

        def can_run(bs: int) -> bool:
            images = [sample.copy() for _ in range(bs)]
            try:
                _ = self._encode_image_batch(images)
                torch.cuda.synchronize()
                return True
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
                    clean_cuda()
                    return False
                raise
            finally:
                for img in images:
                    img.close()

        try:
            low = 1
            best = min(8, requested)
            candidate = max(1, min(requested, max_probe))
            while candidate <= max_probe and can_run(candidate):
                best = candidate
                low = candidate + 1
                candidate *= 2
            high = min(max_probe, candidate - 1 if candidate > best else best)
            while low <= high:
                mid = (low + high) // 2
                if can_run(mid):
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            self._autotuned_batch_size = safe_image_batch_size(self.runtime, best)
            LOGGER.info("Autotuned CLIP image batch size to %d", self._autotuned_batch_size)
            return self._autotuned_batch_size
        finally:
            sample.close()
            clean_cuda()

    def _load_images(self, batch_paths: Sequence[str], max_workers: int) -> List[Image.Image]:
        def _read_one(p: str) -> Image.Image:
            img = Image.open(p).convert("RGB")
            return img.copy()

        worker_count = max(1, min(max_workers, len(batch_paths)))
        if worker_count <= 1:
            return [_read_one(p) for p in batch_paths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as ex:
            return list(ex.map(_read_one, batch_paths))

    def encode_paths(self, paths: Sequence[str], batch_size: int) -> np.ndarray:
        if not paths:
            return np.zeros((0, 512), dtype="float32")

        batch_size = safe_image_batch_size(self.runtime, batch_size)
        if self.device == "cuda":
            batch_size = self._autotune_batch_size(paths, batch_size)

        outputs: List[np.ndarray] = []
        effective_batch = max(1, batch_size)
        worker_count = safe_image_loader_workers(self.device, effective_batch)

        while True:
            try:
                for start_idx in range(0, len(paths), effective_batch):
                    batch_paths = paths[start_idx : start_idx + effective_batch]
                    images = self._load_images(batch_paths, worker_count)
                    try:
                        outputs.append(self._encode_image_batch(images))
                    finally:
                        for image in images:
                            image.close()
                break
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
                    clean_cuda()
                    if effective_batch <= 1:
                        raise
                    new_batch = max(1, effective_batch // 2)
                    LOGGER.warning("CLIP image encoding hit OOM at batch_size=%d. Retrying with batch_size=%d.", effective_batch, new_batch)
                    outputs = []
                    worker_count = safe_image_loader_workers(self.device, new_batch)
                    effective_batch = new_batch
                    continue
                raise
            except Exception as exc:
                msg = str(exc).lower()
                if ("worker" in msg and "exited unexpectedly" in msg) or "oom" in msg or "killed" in msg:
                    if worker_count > 0:
                        LOGGER.warning("Image loader workers failed at workers=%d. Retrying with workers=0 and batch_size=%d.", worker_count, max(1, effective_batch // 2))
                        outputs = []
                        worker_count = 0
                        effective_batch = max(1, effective_batch // 2)
                        clean_cuda()
                        continue
                raise
        return np.vstack(outputs) if outputs else np.zeros((0, 512), dtype="float32")


class SimpleImageEmbedder:
    def __init__(self, bins_per_channel: int = 8, image_size: int = 128) -> None:
        self.bins_per_channel = bins_per_channel
        self.image_size = image_size

    def _encode_one(self, path: str) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        try:
            image = image.resize((self.image_size, self.image_size))
            arr = np.asarray(image, dtype=np.uint8)
            feats: List[np.ndarray] = []
            for channel in range(3):
                hist, _ = np.histogram(arr[:, :, channel], bins=self.bins_per_channel, range=(0, 256), density=True)
                feats.append(hist.astype("float32"))
            vec = np.concatenate(feats, axis=0)
            norm = np.linalg.norm(vec) + 1e-12
            return (vec / norm).astype("float32")
        finally:
            image.close()

    def encode_paths(self, paths: Sequence[str], batch_size: int) -> np.ndarray:
        del batch_size
        max_workers = max(1, min(CONFIG.image_loader_workers, available_cpu_count(), 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            arrs = list(ex.map(self._encode_one, paths))
        return np.vstack(arrs).astype("float32") if arrs else np.zeros((0, self.bins_per_channel * 3), dtype="float32")


class SentenceTransformerTextEmbedder:
    def __init__(self, model_name: str, runtime: Dict[str, Any]) -> None:
        SentenceTransformer = lazy_import_sentence_transformer()
        device = runtime["device"]
        self.runtime = runtime
        self.model = SentenceTransformer(model_name, device=device)
        self._autotuned_batch_size: Optional[int] = None

    def _autotune_batch_size(self, texts: Sequence[str], initial_batch_size: int) -> int:
        requested = safe_text_batch_size(self.runtime, initial_batch_size)
        if self.runtime["device"] != "cuda" or not CONFIG.aggressive_gpu_autotune or self._autotuned_batch_size is not None or not texts:
            return requested if self._autotuned_batch_size is None else self._autotuned_batch_size
        torch = lazy_import_torch()
        max_probe = safe_text_batch_size(self.runtime, max(requested, CONFIG.text_batch_probe_max))
        probe_text = max([str(t) for t in texts[: min(32, len(texts))]] or ["dermoscopy lesion description"], key=len)

        def can_run(bs: int) -> bool:
            try:
                _ = self.model.encode(
                    [probe_text] * bs,
                    batch_size=bs,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                torch.cuda.synchronize()
                return True
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                    clean_cuda()
                    return False
                raise

        low = 1
        best = 1
        candidate = max(1, min(requested, max_probe))
        while candidate <= max_probe and can_run(candidate):
            best = candidate
            low = candidate + 1
            candidate *= 2
        high = min(max_probe, candidate - 1 if candidate > best else best)
        while low <= high:
            mid = (low + high) // 2
            if can_run(mid):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        self._autotuned_batch_size = safe_text_batch_size(self.runtime, best)
        LOGGER.info("Autotuned text batch size to %d", best)
        clean_cuda()
        return best

    def encode_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        effective_batch = self._autotune_batch_size(texts, batch_size)
        while True:
            try:
                emb = self.model.encode(
                    list(texts),
                    batch_size=effective_batch,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                return np.asarray(emb, dtype="float32")
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
                    clean_cuda()
                    if effective_batch <= 1:
                        raise
                    new_batch = max(1, effective_batch // 2)
                    LOGGER.warning("Text embedding hit OOM at batch_size=%d. Retrying with batch_size=%d.", effective_batch, new_batch)
                    effective_batch = new_batch
                    continue
                raise


class SimpleTextEmbedder:
    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), stop_words="english")
        self._is_fit = False

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        mat = self.vectorizer.fit_transform(list(texts))
        self._is_fit = True
        return mat.astype(np.float32).toarray()

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("SimpleTextEmbedder must be fit before transform().")
        return self.vectorizer.transform(list(texts)).astype(np.float32).toarray()


# -----------------------------------------------------------------------------
# Index wrappers
# -----------------------------------------------------------------------------


class ImageIndex:
    def __init__(self, backend: str, runtime: Dict[str, Any]) -> None:
        self.backend = backend
        self.runtime = runtime
        self.index = SimilarityIndex()
        self.embeddings: Optional[np.ndarray] = None
        self.meta: Optional[pd.DataFrame] = None
        if backend == "clip":
            try:
                self.embedder = ClipImageEmbedder(CONFIG.clip_model_name, runtime)
            except Exception as exc:
                LOGGER.warning("Failed to initialize CLIP image backend (%s). Falling back to simple image features.", exc)
                self.backend = "simple"
                self.embedder = SimpleImageEmbedder(image_size=CONFIG.simple_image_size)
        elif backend == "simple":
            self.embedder = SimpleImageEmbedder(image_size=CONFIG.simple_image_size)
        else:
            raise ValueError(f"Unsupported image backend: {backend}")

    def encode_paths(self, paths: Sequence[str]) -> np.ndarray:
        return self.embedder.encode_paths(paths, batch_size=tuned_image_batch_size(self.runtime))

    def build(self, df: pd.DataFrame, embedding_cache_path: Optional[Path] = None) -> None:
        paths = df["image_path"].astype(str).tolist()
        if embedding_cache_path is not None and embedding_cache_path.exists():
            feats = np.load(embedding_cache_path)
        else:
            feats = self.encode_paths(paths)
            if embedding_cache_path is not None:
                np.save(embedding_cache_path, feats)
        self.embeddings = np.asarray(feats, dtype="float32")
        self.meta = df.reset_index(drop=True).copy()
        self.index.build(self.embeddings, self.meta)

    def search_by_vector(self, query_vec: np.ndarray, top_k: int) -> pd.DataFrame:
        return self.index.search_by_vector(query_vec, top_k)

    def search_many_by_vectors(self, query_vecs: np.ndarray, top_k: int) -> List[pd.DataFrame]:
        return self.index.search_many_by_vectors(query_vecs, top_k)

    def save(self, prefix: Path) -> None:
        if self.meta is None or self.embeddings is None:
            raise RuntimeError("Nothing to save.")
        self.meta.to_csv(prefix.with_suffix(".csv"), index=False)
        np.save(prefix.with_suffix(".npy"), self.embeddings)


class TextChunkIndex:
    def __init__(self, backend: str, runtime: Dict[str, Any]) -> None:
        self.backend = backend
        self.runtime = runtime
        self.index = SimilarityIndex()
        self.meta: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        if backend == "sentence_transformer":
            try:
                self.embedder = SentenceTransformerTextEmbedder(CONFIG.text_model_name, runtime)
            except Exception as exc:
                LOGGER.warning("Failed to initialize sentence-transformer backend (%s). Falling back to simple TF-IDF text features.", exc)
                self.backend = "simple"
                self.embedder = SimpleTextEmbedder()
        elif backend == "simple":
            self.embedder = SimpleTextEmbedder()
        else:
            raise ValueError(f"Unsupported text backend: {backend}")

    def build(self, chunk_df: pd.DataFrame, embedding_cache_path: Optional[Path] = None) -> None:
        texts = chunk_df["chunk_text"].astype(str).tolist()
        if embedding_cache_path is not None and embedding_cache_path.exists():
            emb = np.load(embedding_cache_path)
        else:
            if self.backend == "sentence_transformer":
                emb = self.embedder.encode_texts(texts, batch_size=tuned_text_batch_size(self.runtime))
            else:
                emb = self.embedder.fit_transform(texts)
                norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
                emb = (emb / norms).astype("float32")
            if embedding_cache_path is not None:
                np.save(embedding_cache_path, emb)
        self.embeddings = np.asarray(emb, dtype="float32")
        self.meta = chunk_df.reset_index(drop=True).copy()
        self.index.build(self.embeddings, self.meta)

    def search(self, query: str, top_k: int) -> pd.DataFrame:
        if not str(query).strip() or self.meta is None:
            return pd.DataFrame(columns=["chunk_id", "source_id", "chunk_text", "similarity"])
        if self.backend == "sentence_transformer":
            q = self.embedder.encode_texts([query], batch_size=1)[0]
        else:
            q = self.embedder.transform([query])[0]
            q = q / (np.linalg.norm(q) + 1e-12)
        return self.index.search_by_vector(np.asarray(q, dtype="float32"), top_k)


# -----------------------------------------------------------------------------
# Retrieval evidence and prompting
# -----------------------------------------------------------------------------


def build_patient_context(row: pd.Series) -> str:
    parts = []
    for key, label in [("age_approx", "Approx age"), ("sex", "Sex"), ("anatom_site_general", "Anatomical site")]:
        if key in row.index and pd.notna(row[key]) and str(row[key]).strip():
            parts.append(f"{label}: {row[key]}")
    for key in ["patient_text", "notes", "symptoms", "history"]:
        if key in row.index and pd.notna(row[key]) and str(row[key]).strip():
            parts.append(f"Patient info: {row[key]}")
    return "\n".join(parts)


def render_case_evidence(neighbors: pd.DataFrame, targets: Sequence[str]) -> str:
    if neighbors.empty:
        return "No retrieved similar cases."
    lines = []
    for i, (_, row) in enumerate(neighbors.iterrows(), start=1):
        fields = [f"Case {i}: isic_id={row.get('isic_id')}", f"similarity={row.get('similarity', 0):.4f}"]
        for target in targets:
            value = row.get(target)
            if pd.notna(value) and str(value).strip():
                fields.append(f"{target}={value}")
        lines.append(", ".join(fields))
    return "\n".join(lines)


def build_text_query(row: pd.Series, neighbors: pd.DataFrame, targets: Sequence[str]) -> str:
    parts = []
    patient = build_patient_context(row)
    if patient:
        parts.append(patient)
    for target in targets:
        vals = [str(x).strip() for x in neighbors.get(target, pd.Series(dtype=str)).dropna().tolist() if str(x).strip()]
        if vals:
            common = pd.Series(vals).value_counts().head(3).index.tolist()
            parts.append(f"{target}: " + ", ".join(common))
    return "\n".join(parts).strip()


def build_response_schema(vocab: Dict[str, List[str]], targets: Sequence[str], include_extras: bool = True) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    required: List[str] = []
    for target in targets:
        enums = vocab.get(target, [])
        props[target] = {"type": ["string", "null"], "enum": enums + [None]}
        required.append(target)
    if include_extras:
        props["rationale"] = {"type": ["string", "null"]}
        props["needs_review"] = {"type": ["boolean", "null"]}
        required.extend(["rationale", "needs_review"])
    return {"type": "object", "properties": props, "required": required}


def prompt_label_block(vocab: Dict[str, List[str]], targets: Sequence[str]) -> str:
    return "\n".join(f"{target}: {', '.join(vocab.get(target, [])[:80])}" for target in targets)


def render_prior_vlm_context(vlm_pred: Optional[Dict[str, Any]], targets: Sequence[str]) -> str:
    if not vlm_pred:
        return "No prior VLM prediction available."
    lines = [f"status: {vlm_pred.get('status', 'unknown')}"]
    for target in targets:
        value = vlm_pred.get(target)
        lines.append(f"{target}: {value if value is not None and str(value).strip() else 'null'}")
    err = vlm_pred.get("error_message")
    if err:
        lines.append(f"error_message: {err}")
    review = vlm_pred.get("needs_review")
    if review is not None:
        lines.append(f"needs_review: {review}")
    return "\n".join(lines)


def build_llm_prompt(
    row: pd.Series,
    neighbors: pd.DataFrame,
    text_chunks: pd.DataFrame,
    vocab: Dict[str, List[str]],
    targets: Sequence[str],
    vlm_pred: Optional[Dict[str, Any]] = None,
) -> str:
    med_text = (
        "\n".join(f"- {txt}" for txt in text_chunks.get("chunk_text", pd.Series(dtype=str)).astype(str).tolist())
        if not text_chunks.empty
        else "No external medical text retrieved."
    )
    return textwrap.dedent(
        f"""
        You are assisting with dermoscopic lesion classification.

        Predict ONLY the following target fields:
        {", ".join(targets)}

        Allowed labels:
        {prompt_label_block(vocab, targets)}

        Patient context:
        {build_patient_context(row) or "None"}

        Retrieved visually similar cases:
        {render_case_evidence(neighbors, targets)}

        Prior VLM prediction for this same sample:
        {render_prior_vlm_context(vlm_pred, targets)}
        Treat the VLM prediction as auxiliary evidence, not as ground truth.

        Retrieved medical text:
        {med_text}

        Return only JSON matching the schema.
        If uncertain for a field, use null.
        """
    ).strip()


def build_vlm_prompt(
    row: pd.Series,
    neighbors: pd.DataFrame,
    vocab: Dict[str, List[str]],
    targets: Sequence[str],
) -> str:
    return textwrap.dedent(
        f"""
        You are assisting with dermoscopic lesion classification from an image.

        Predict ONLY these target fields and nothing else:
        {", ".join(targets)}

        Allowed labels:
        {prompt_label_block(vocab, targets)}

        Patient context:
        {build_patient_context(row) or "None"}

        Retrieved visually similar cases:
        {render_case_evidence(neighbors, targets)}

        Use the image plus similar-case evidence.
        Do NOT use any external medical text in this branch.
        Return ONLY a single compact JSON object with exactly these target keys.
        Do not include rationale, markdown, code fences, or any extra text.
        If uncertain for a field, use null.
        """
    ).strip()


def sanitize_prediction_dict(pred: Dict[str, Any], vocab: Dict[str, List[str]], targets: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for target in targets:
        value = pred.get(target)
        if value is None:
            out[target] = None
        else:
            s = str(value).strip()
            out[target] = s if s in set(vocab.get(target, [])) else None
    out["rationale"] = pred.get("rationale")
    out["needs_review"] = pred.get("needs_review")
    return out


# -----------------------------------------------------------------------------
# Local Llama / Ollama helpers
# -----------------------------------------------------------------------------


SPAWNED_OLLAMA_PROC: Optional[subprocess.Popen] = None


def cleanup_spawned_ollama() -> None:
    global SPAWNED_OLLAMA_PROC
    proc = SPAWNED_OLLAMA_PROC
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
    except Exception:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
    finally:
        SPAWNED_OLLAMA_PROC = None


atexit.register(cleanup_spawned_ollama)


def _ollama_tags_url(args: argparse.Namespace) -> str:
    return f"{args.ollama_base_url.rstrip('/')}" + "/api/tags"


def _ollama_chat_url(args: argparse.Namespace) -> str:
    return f"{args.ollama_base_url.rstrip('/')}" + "/api/chat"


def _parse_ollama_url(url: str):
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    return parsed, host, port


def ollama_is_reachable(args: argparse.Namespace, timeout: int = 10) -> bool:
    try:
        r = requests.get(_ollama_tags_url(args), timeout=timeout)
        r.raise_for_status()
        return True
    except Exception:
        return False


def require_ollama_running(args: argparse.Namespace) -> None:
    if not ollama_is_reachable(args, timeout=10):
        raise RuntimeError(f"Ollama is not reachable at {args.ollama_base_url}")


def _looks_like_local_ollama_url(url: str) -> bool:
    parsed, host, _ = _parse_ollama_url(url)
    return (parsed.scheme or "http") == "http" and host in {"127.0.0.1", "localhost", "0.0.0.0"}


def _detect_ollama_download_url() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        arch = "amd64"
    elif machine in {"aarch64", "arm64"}:
        arch = "arm64"
    else:
        raise RuntimeError(f"Unsupported architecture for automatic Ollama installation: {machine}")
    return f"https://ollama.com/download/ollama-linux-{arch}.tar.zst"


def _candidate_ollama_binaries(args: Optional[argparse.Namespace] = None) -> List[Path]:
    cands: List[Path] = []
    if args is not None:
        install_dir = Path(args.ollama_install_dir).resolve()
        cands.extend([install_dir / "bin" / "ollama", install_dir / "usr" / "bin" / "ollama"])
    path_bin = shutil.which("ollama")
    if path_bin:
        cands.append(Path(path_bin))
    uniq: List[Path] = []
    seen = set()
    for p in cands:
        rp = p.resolve() if p.exists() else p
        key = str(rp)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _prepend_env_path(name: str, value: Path) -> None:
    value = value.resolve()
    current = os.environ.get(name, "")
    parts = [p for p in current.split(os.pathsep) if p]
    if str(value) not in parts:
        os.environ[name] = os.pathsep.join([str(value)] + parts) if parts else str(value)


def prepare_ollama_runtime_env(args: argparse.Namespace) -> None:
    args.ollama_install_dir = Path(args.ollama_install_dir).resolve()
    args.ollama_models_dir = Path(args.ollama_models_dir).resolve()
    args.ollama_tmp_dir = Path(args.ollama_tmp_dir).resolve()
    args.ollama_models_dir.mkdir(parents=True, exist_ok=True)
    args.ollama_tmp_dir.mkdir(parents=True, exist_ok=True)
    install_dir = Path(args.ollama_install_dir)
    bin_dir = install_dir / "bin"
    lib_dir = install_dir / "lib"
    runners_dir = lib_dir / "ollama"
    if bin_dir.exists():
        _prepend_env_path("PATH", bin_dir)
    if runners_dir.exists():
        _prepend_env_path("LD_LIBRARY_PATH", runners_dir)
    if lib_dir.exists():
        _prepend_env_path("LD_LIBRARY_PATH", lib_dir)
    os.environ.setdefault("OLLAMA_MODELS", str(args.ollama_models_dir))
    os.environ.setdefault("OLLAMA_TMPDIR", str(args.ollama_tmp_dir))


def _ensure_zstandard() -> Any:
    try:
        import zstandard as zstd  # type: ignore
        return zstd
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "zstandard"], check=True)
        import zstandard as zstd  # type: ignore
        return zstd


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tf.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f"Unsafe tar member during Ollama installation: {member.name}")
    tf.extractall(dest)


def maybe_install_ollama(args: argparse.Namespace) -> Optional[str]:
    prepare_ollama_runtime_env(args)
    existing = _ollama_binary(args)
    if existing:
        return existing
    if not args.auto_install_ollama:
        return None
    if not _looks_like_local_ollama_url(args.ollama_base_url):
        LOGGER.warning("Cannot auto-install Ollama because the configured base URL is not local: %s", args.ollama_base_url)
        return None

    install_dir = Path(args.ollama_install_dir).resolve()
    install_dir.mkdir(parents=True, exist_ok=True)
    url = (args.ollama_install_url or "").strip() or _detect_ollama_download_url()
    LOGGER.info("Installing local Ollama runtime into %s from %s", install_dir, url)
    tmp_root = Path(tempfile.mkdtemp(prefix="ollama_install_"))
    archive_zst = tmp_root / "ollama.tar.zst"
    archive_tar = tmp_root / "ollama.tar"
    try:
        urllib.request.urlretrieve(url, archive_zst)
        zstd = _ensure_zstandard()
        with open(archive_zst, "rb") as src_f, open(archive_tar, "wb") as dst_f:
            dctx = zstd.ZstdDecompressor()
            dctx.copy_stream(src_f, dst_f)
        with tarfile.open(archive_tar, "r:") as tf:
            _safe_extract_tar(tf, install_dir)
        for cand in _candidate_ollama_binaries(args):
            if cand.exists():
                try:
                    cand.chmod(cand.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                except Exception:
                    pass
                prepare_ollama_runtime_env(args)
                return str(cand)
        raise RuntimeError(f"Ollama installation finished but binary was not found under {install_dir}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _ollama_binary(args: Optional[argparse.Namespace] = None) -> Optional[str]:
    for cand in _candidate_ollama_binaries(args):
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand.resolve())
    return shutil.which("ollama")


def _ollama_log_path() -> Path:
    return Path.cwd() / "ollama_server.log"


def _ollama_diag(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {"base_url": args.ollama_base_url, "binary": _ollama_binary(args), "reachable": ollama_is_reachable(args, timeout=3)}
    if out["binary"]:
        env = os.environ.copy()
        parsed, host, port = _parse_ollama_url(args.ollama_base_url)
        env.setdefault("OLLAMA_HOST", f"{host}:{port}")
        for cmd_name in ([out["binary"], "--version"], [out["binary"], "list"], [out["binary"], "ps"]):
            try:
                cp = subprocess.run(cmd_name, capture_output=True, text=True, timeout=20, env=env)
                out[" ".join(cmd_name[1:])] = {"rc": cp.returncode, "stdout": cp.stdout[-4000:], "stderr": cp.stderr[-4000:]}
            except Exception as exc:
                out[" ".join(cmd_name[1:])] = {"error": str(exc)}
    log_path = _ollama_log_path()
    if log_path.exists():
        try:
            out["server_log_tail"] = log_path.read_text(encoding="utf-8", errors="ignore")[-8000:]
        except Exception as exc:
            out["server_log_tail_error"] = str(exc)
    return out


def maybe_start_ollama_server(args: argparse.Namespace) -> bool:
    global SPAWNED_OLLAMA_PROC
    prepare_ollama_runtime_env(args)
    if ollama_is_reachable(args, timeout=3):
        return True
    if not args.auto_start_ollama:
        return False
    if not _looks_like_local_ollama_url(args.ollama_base_url):
        LOGGER.warning("Not auto-starting Ollama because the base URL is not local: %s", args.ollama_base_url)
        return False
    ollama_bin = _ollama_binary(args) or maybe_install_ollama(args)
    if not ollama_bin:
        LOGGER.warning("Cannot auto-start Ollama because the 'ollama' binary is not installed and local installation failed.")
        return False

    parsed, host, port = _parse_ollama_url(args.ollama_base_url)
    env = os.environ.copy()
    env.setdefault("OLLAMA_HOST", f"{host}:{port}")

    log_hint = f"OLLAMA_HOST={env['OLLAMA_HOST']}"
    log_path = _ollama_log_path()
    env.setdefault("OLLAMA_TMPDIR", str((Path.cwd() / ".ollama_tmp").resolve()))
    Path(env["OLLAMA_TMPDIR"]).mkdir(parents=True, exist_ok=True)
    LOGGER.info("Attempting to start local Ollama server via `%s serve` (%s). Logs: %s", ollama_bin, log_hint, log_path)
    try:
        log_f = open(log_path, "a", encoding="utf-8")
        SPAWNED_OLLAMA_PROC = subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
    except Exception as exc:
        LOGGER.warning("Failed to spawn Ollama server: %s", exc)
        SPAWNED_OLLAMA_PROC = None
        return False

    deadline = time.time() + CONFIG.ollama_start_timeout_seconds
    while time.time() < deadline:
        proc = SPAWNED_OLLAMA_PROC
        if proc is not None and proc.poll() is not None:
            LOGGER.warning("Spawned Ollama server exited early with return code %s.", proc.returncode)
            SPAWNED_OLLAMA_PROC = None
            return False
        if ollama_is_reachable(args, timeout=3):
            return True
        time.sleep(CONFIG.ollama_healthcheck_interval_seconds)
    LOGGER.warning("Timed out waiting for Ollama to become healthy at %s.", args.ollama_base_url)
    return False


def _ollama_registry_diag() -> Dict[str, Any]:
    diag: Dict[str, Any] = {}
    for url in ["https://registry.ollama.ai/v2/", "https://ollama.com/download/"]:
        try:
            r = requests.get(url, timeout=20)
            diag[url] = {"ok": True, "status": r.status_code}
        except Exception as exc:
            diag[url] = {"ok": False, "error": str(exc)}
    return diag


def maybe_pull_ollama_model(args: argparse.Namespace, model_name: str) -> bool:
    prepare_ollama_runtime_env(args)
    if not args.auto_pull_ollama_models:
        return False
    if not _looks_like_local_ollama_url(args.ollama_base_url):
        LOGGER.warning("Cannot auto-pull model %s because the configured Ollama URL is not local: %s", model_name, args.ollama_base_url)
        return False
    ollama_bin = _ollama_binary(args)
    if not ollama_bin:
        LOGGER.warning("Cannot auto-pull model %s because the 'ollama' binary is unavailable.", model_name)
        return False

    parsed, host, port = _parse_ollama_url(args.ollama_base_url)
    env = os.environ.copy()
    env.setdefault("OLLAMA_HOST", f"{host}:{port}")
    env.setdefault("GODEBUG", "http2client=0")
    retries = max(1, int(os.environ.get("DERM_OLLAMA_PULL_RETRIES", "5")))
    last_err = None

    for attempt in range(1, retries + 1):
        LOGGER.info("Pulling missing Ollama model %s (attempt %d/%d)", model_name, attempt, retries)
        try:
            cp = subprocess.run(
                [ollama_bin, "pull", model_name],
                check=False,
                env=env,
                timeout=CONFIG.ollama_pull_timeout_seconds,
                capture_output=True,
                text=True,
            )
            if cp.returncode == 0:
                return True
            last_err = f"returncode={cp.returncode}; stdout={cp.stdout[-2000:]}; stderr={cp.stderr[-2000:]}"
            LOGGER.warning("Failed to pull Ollama model %s on attempt %d/%d: %s", model_name, attempt, retries, last_err)
        except Exception as exc:
            last_err = str(exc)
            LOGGER.warning("Failed to pull Ollama model %s on attempt %d/%d: %s", model_name, attempt, retries, exc)
        if attempt < retries:
            time.sleep(min(60, 5 * attempt))
    LOGGER.error(
        "Exhausted retries pulling Ollama model %s. Registry diagnostics: %s. Last error: %s",
        model_name,
        json.dumps(_ollama_registry_diag(), ensure_ascii=False),
        last_err,
    )
    return False


def _required_ollama_models_for_branches(args: argparse.Namespace) -> Dict[str, str]:
    required: Dict[str, str] = {}
    if "llm_rag" in args.branches and not args.disable_local_llm:
        required["llm_rag"] = args.ollama_llm_model
    if "vlm_rag" in args.branches and not args.disable_local_llm:
        required["vlm_rag"] = args.ollama_vlm_model
    return required


def _encode_image_file_as_png_base64(path: str) -> str:
    img_path = Path(path)
    if not img_path.exists():
        raise RuntimeError(f"Image file not found for Ollama request: {img_path}")
    with Image.open(img_path) as img:
        rgb = img.convert("RGB")
        buf = BytesIO()
        rgb.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pick_vlm_warmup_image(args: argparse.Namespace) -> str:
    images_root = image_root(Path(args.data_dir))
    if not images_root.exists():
        raise RuntimeError(f"VLM warmup images folder not found: {images_root}")
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            try:
                with Image.open(p) as img:
                    img.verify()
                return str(p)
            except Exception:
                continue
    raise RuntimeError(f"No valid warmup image found under {images_root}")


def warmup_ollama_model(args: argparse.Namespace, model_name: str, *, vision: bool) -> None:
    prepare_ollama_runtime_env(args)
    message: Dict[str, Any] = {"role": "user", "content": 'Return exactly {"ok": true}.'}
    options: Dict[str, Any] = {"temperature": 0}
    if vision:
        warmup_image = _pick_vlm_warmup_image(args)
        message["images"] = [_encode_image_file_as_png_base64(warmup_image)]
        options["num_ctx"] = 8192
    payload = {
        "model": model_name,
        "messages": [message],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
        "options": options,
        "keep_alive": args.ollama_keep_alive,
    }
    r = requests.post(_ollama_chat_url(args), json=payload, timeout=900)
    r.raise_for_status()
    data = r.json()
    raw = str(data.get("message", {}).get("content", "")).strip()
    if not raw:
        raise RuntimeError(f"Warmup returned empty content for model {model_name}: {data}")


def ensure_requested_ollama_models(args: argparse.Namespace) -> Tuple[set, Dict[str, str]]:
    names = available_ollama_models(args)
    required_by_branch = _required_ollama_models_for_branches(args)
    missing = {branch: model for branch, model in required_by_branch.items() if model not in names}
    if missing and args.auto_pull_ollama_models:
        for model_name in sorted(set(missing.values())):
            if maybe_pull_ollama_model(args, model_name):
                names = available_ollama_models(args)
        missing = {branch: model for branch, model in required_by_branch.items() if model not in names}
    return names, missing


def list_ollama_models(args: argparse.Namespace) -> List[str]:
    r = requests.get(_ollama_tags_url(args), timeout=20)
    r.raise_for_status()
    return [m.get("name") for m in r.json().get("models", []) if m.get("name")]


def available_ollama_models(args: argparse.Namespace) -> set:
    try:
        return set(list_ollama_models(args))
    except Exception:
        return set()


def ensure_ollama_models_present(args: argparse.Namespace) -> None:
    names, missing_by_branch = ensure_requested_ollama_models(args)
    missing = sorted(set(missing_by_branch.values()))
    if missing:
        raise RuntimeError(f"Missing Ollama models: {missing}")


def file_hash(path: Optional[str]) -> str:
    if not path:
        return "no_image"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:20]


def prediction_cache_key(branch: str, image_id: str, prompt: str, model_name: str, image_path: Optional[str]) -> str:
    payload = json.dumps(
        {
            "branch": branch,
            "image_id": image_id,
            "prompt": prompt,
            "model_name": model_name,
            "image_hash": file_hash(image_path),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def image_to_base64(path: str) -> str:
    return _encode_image_file_as_png_base64(path)

def _strip_json_code_fences(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _extract_first_json_object(raw_text: str) -> Optional[str]:
    text = _strip_json_code_fences(raw_text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def _coerce_scalar_token(token: str) -> Any:
    token = token.strip()
    if not token:
        return None
    low = token.lower()
    if low == 'null':
        return None
    if low == 'true':
        return True
    if low == 'false':
        return False
    if token.startswith('"') and token.endswith('"') and len(token) >= 2:
        return token[1:-1].replace('\"', '"').strip()
    return token.strip().strip('"').strip()


def _salvage_prediction_from_text(raw_text: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = _strip_json_code_fences(raw_text)
    props = schema.get('properties', {})
    salvaged: Dict[str, Any] = {}
    found_any = False
    for key, spec in props.items():
        pattern = re.compile(rf"[\"']?{re.escape(key)}[\"']?\s*:\s*(.+?)(?=,\s*[\"']?[A-Za-z0-9_]+[\"']?\s*:|\s*}}|$)", re.DOTALL)
        match = pattern.search(text)
        if not match:
            continue
        token = match.group(1).strip()
        value = _coerce_scalar_token(token)
        enums = spec.get('enum') or []
        if enums:
            enum_strings = {str(x): x for x in enums if x is not None}
            if value is None:
                salvaged[key] = None
                found_any = True
            else:
                s = str(value).strip()
                if s in enum_strings:
                    salvaged[key] = s
                    found_any = True
                else:
                    for allowed in enum_strings:
                        if allowed.lower() == s.lower():
                            salvaged[key] = allowed
                            found_any = True
                            break
        else:
            salvaged[key] = value
            found_any = True
    return salvaged if found_any else None


def _parse_ollama_structured_response(raw_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    text = _strip_json_code_fences(raw_text)
    try:
        return json.loads(text)
    except Exception:
        pass
    obj = _extract_first_json_object(text)
    if obj:
        try:
            return json.loads(obj)
        except Exception:
            pass
    salvaged = _salvage_prediction_from_text(text, schema)
    if salvaged is not None:
        return salvaged
    raise json.JSONDecodeError('Could not parse structured Ollama JSON', text, 0)


def call_ollama_json(
    args: argparse.Namespace,
    model_name: str,
    prompt: str,
    schema: Dict[str, Any],
    branch: str,
    image_id: str,
    cache_dir: Path,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    cache_key = prediction_cache_key(branch, image_id, prompt, model_name, image_path)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    message: Dict[str, Any] = {"role": "user", "content": prompt}
    if image_path:
        message["images"] = [image_to_base64(image_path)]

    options: Dict[str, Any] = {"temperature": 0.0}
    if branch == "vlm_rag" or image_path:
        options.update({"top_p": 0.1, "num_predict": 256, "num_ctx": 8192})
    else:
        options.update({"num_predict": 384, "num_ctx": 8192})

    payload = {
        "model": model_name,
        "messages": [message],
        "stream": False,
        "format": schema,
        "options": options,
        "keep_alive": args.ollama_keep_alive,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1, CONFIG.api_max_retries + 1):
        try:
            r = requests.post(_ollama_chat_url(args), json=payload, timeout=1800)
            r.raise_for_status()
            data = r.json()
            raw_text = data.get("message", {}).get("content", "")
            if not raw_text or not str(raw_text).strip():
                raise RuntimeError(f"Ollama returned empty content. Raw={data}")
            parsed = _parse_ollama_structured_response(str(raw_text), schema)
            cache_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
            return parsed
        except Exception as exc:
            last_exc = exc
            LOGGER.warning("Ollama call attempt %s/%s failed for model %s (%s): %s", attempt, CONFIG.api_max_retries, model_name, branch, exc)
            try:
                if not ollama_is_reachable(args, timeout=3):
                    maybe_start_ollama_server(args)
                names, _ = ensure_requested_ollama_models(args)
                if model_name not in names and args.auto_pull_ollama_models:
                    maybe_pull_ollama_model(args, model_name)
            except Exception as repair_exc:
                LOGGER.warning("Ollama repair attempt failed: %s", repair_exc)
            if attempt < CONFIG.api_max_retries:
                time.sleep(CONFIG.api_retry_sleep * attempt)
            else:
                raise RuntimeError(f"Ollama call failed after retries: {exc}. Diagnostics: {json.dumps(_ollama_diag(args), ensure_ascii=False)[:4000]}") from exc
    raise RuntimeError(str(last_exc))


# -----------------------------------------------------------------------------
# Prediction and evaluation
# -----------------------------------------------------------------------------


def weighted_knn_predict(neighbors: pd.DataFrame, targets: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if neighbors.empty:
        for target in targets:
            out[target] = None
        out["rationale"] = "No retrieved neighbors."
        out["needs_review"] = True
        return out

    for target in targets:
        scores: Dict[str, float] = {}
        for _, row in neighbors.iterrows():
            label = row.get(target)
            if pd.isna(label) or not str(label).strip():
                continue
            scores[str(label)] = scores.get(str(label), 0.0) + float(row.get("similarity", 0.0))
        out[target] = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None

    out["rationale"] = "Similarity-weighted kNN over retrieved train cases."
    out["needs_review"] = False
    return out


def make_error_prediction(targets: Sequence[str], error_message: str) -> Dict[str, Any]:
    out = {target: None for target in targets}
    out["rationale"] = None
    out["needs_review"] = True
    out["status"] = "error"
    out["error_message"] = error_message
    return out


def flatten_prediction_row(row: pd.Series, branch: str, pred: Dict[str, Any], targets: Sequence[str]) -> Dict[str, Any]:
    out = {
        "isic_id": row["isic_id"],
        "branch": branch,
        "image_path": row["image_path"],
        "status": pred.get("status", "ok"),
        "error_message": pred.get("error_message"),
        "rationale": pred.get("rationale"),
        "needs_review": pred.get("needs_review"),
    }
    for target in targets:
        out[f"true__{target}"] = row.get(target)
        out[f"pred__{target}"] = pred.get(target)
    return out


def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(df["isic_id"].astype(str).tolist()) if "isic_id" in df.columns else set()




def load_branch_prediction_map(path: Path | str, targets: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Load per-sample branch predictions from a CSV written by run_branch_predictions.

    Returns a map keyed by isic_id with compact structured evidence that can be injected
    into later prompts, e.g. VLM outputs passed into the LLM branch. Missing files or empty
    files return an empty mapping instead of raising.
    """
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        LOGGER.warning("Could not read branch prediction map from %s: %s", csv_path, exc)
        return {}

    if df.empty or "isic_id" not in df.columns:
        return {}

    mapping: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        isic_id = str(row.get("isic_id", "")).strip()
        if not isic_id:
            continue

        entry: Dict[str, Any] = {
            "status": None if pd.isna(row.get("status")) else row.get("status"),
            "error_message": None if pd.isna(row.get("error_message")) else row.get("error_message"),
            "needs_review": None if pd.isna(row.get("needs_review")) else row.get("needs_review"),
        }
        for target in targets:
            pred_col = f"pred__{target}"
            value = row.get(pred_col) if pred_col in row.index else None
            entry[target] = None if pd.isna(value) else value
        mapping[isic_id] = entry

    return mapping

def _progress_log_interval(total_rows: int) -> int:
    if total_rows <= 0:
        return 1
    # log roughly every 1% of branch progress, but never more often than every row for tiny sets
    return max(1, total_rows // 100)


def log_branch_progress(
    branch: str,
    total_rows: int,
    completed_rows: int,
    start_time: float,
    *,
    force: bool = False,
) -> None:
    if total_rows <= 0:
        return
    pct = 100.0 * completed_rows / total_rows
    elapsed = max(0.0, time.time() - start_time)
    rate = completed_rows / elapsed if elapsed > 0 else 0.0
    eta_seconds = (total_rows - completed_rows) / rate if rate > 0 else float('inf')
    eta_str = (
        f"{eta_seconds / 60.0:.1f} min"
        if math.isfinite(eta_seconds) and eta_seconds < 7200
        else (f"{eta_seconds / 3600.0:.2f} h" if math.isfinite(eta_seconds) else "unknown")
    )
    LOGGER.info(
        "%s progress: %.1f%% (%s/%s) elapsed=%s eta=%s",
        branch,
        pct,
        completed_rows,
        total_rows,
        f"{elapsed / 60.0:.1f} min" if elapsed < 7200 else f"{elapsed / 3600.0:.2f} h",
        eta_str,
    )


def run_branch_predictions(
    branch: str,
    test_df: pd.DataFrame,
    targets: Sequence[str],
    vocab: Dict[str, List[str]],
    image_index: ImageIndex,
    test_embeddings: np.ndarray,
    text_index: Optional[TextChunkIndex],
    args: argparse.Namespace,
    paths: RuntimePaths,
    neighbor_cache: Optional[List[pd.DataFrame]] = None,
    prior_context_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Path, Path]:
    out_path = paths.predictions_dir / f"{branch}_full_predictions.csv"
    err_path = paths.predictions_dir / f"{branch}_full_errors.csv"
    done_ids = load_done_ids(out_path)
    schema = build_response_schema(vocab, targets, include_extras=(branch != "vlm_rag"))

    if neighbor_cache is None:
        neighbor_cache = image_index.search_many_by_vectors(test_embeddings, top_k=CONFIG.top_k_images)

    total_rows = len(test_df)
    completed_rows = min(len(done_ids), total_rows)
    start_time = time.time()
    progress_interval = _progress_log_interval(total_rows)
    if branch in {"llm_rag", "vlm_rag"}:
        LOGGER.info(
            "%s pipeline starting: total_rows=%s, already_completed=%s, remaining=%s",
            branch,
            total_rows,
            completed_rows,
            max(0, total_rows - completed_rows),
        )
        log_branch_progress(branch, total_rows, completed_rows, start_time, force=True)

    for idx, row in test_df.iterrows():
        isic_id = str(row["isic_id"])
        if isic_id in done_ids:
            continue
        try:
            neighbors = neighbor_cache[idx]
            if branch == "knn":
                pred = weighted_knn_predict(neighbors, targets)
                pred["status"] = "ok"
                pred["error_message"] = None
            elif branch == "llm_rag":
                text_query = build_text_query(row, neighbors, targets)
                text_chunks = text_index.search(text_query, CONFIG.top_k_text_chunks) if text_index is not None else pd.DataFrame()
                vlm_pred = prior_context_map.get(isic_id) if prior_context_map else None
                raw = call_ollama_json(
                    args=args,
                    model_name=args.ollama_llm_model,
                    prompt=build_llm_prompt(row, neighbors, text_chunks, vocab, targets, vlm_pred=vlm_pred),
                    schema=schema,
                    branch=branch,
                    image_id=isic_id,
                    cache_dir=paths.cache_dir,
                    image_path=None,
                )
                pred = sanitize_prediction_dict(raw, vocab, targets)
                pred["status"] = "ok"
                pred["error_message"] = None
                time.sleep(CONFIG.api_inter_request_sleep)
            elif branch == "vlm_rag":
                raw = call_ollama_json(
                    args=args,
                    model_name=args.ollama_vlm_model,
                    prompt=build_vlm_prompt(row, neighbors, vocab, targets),
                    schema=schema,
                    branch=branch,
                    image_id=isic_id,
                    cache_dir=paths.cache_dir,
                    image_path=str(row["image_path"]),
                )
                pred = sanitize_prediction_dict(raw, vocab, targets)
                pred["status"] = "ok"
                pred["error_message"] = None
                time.sleep(CONFIG.api_inter_request_sleep)
            else:
                raise ValueError(f"Unknown branch: {branch}")
            append_csv_row(out_path, flatten_prediction_row(row, branch, pred, targets))
            completed_rows += 1
            if branch in {"llm_rag", "vlm_rag"} and (completed_rows % progress_interval == 0 or completed_rows == total_rows):
                log_branch_progress(branch, total_rows, completed_rows, start_time)
        except Exception as exc:
            append_csv_row(err_path, {"isic_id": isic_id, "branch": branch, "error": str(exc)})
            append_csv_row(out_path, flatten_prediction_row(row, branch, make_error_prediction(targets, str(exc)), targets))
            completed_rows += 1
            if branch in {"llm_rag", "vlm_rag"}:
                LOGGER.warning("%s continuing after prediction error for %s: %s", branch, isic_id, exc)
                log_branch_progress(branch, total_rows, completed_rows, start_time, force=True)
                continue
    if branch in {"llm_rag", "vlm_rag"}:
        log_branch_progress(branch, total_rows, completed_rows, start_time, force=True)
    return out_path, err_path


def evaluate_predictions(
    pred_path: Path,
    branch: str,
    targets: Sequence[str],
    paths: RuntimePaths,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not pred_path.exists():
        return [], {}
    df = pd.read_csv(pred_path)
    compact_rows: List[Dict[str, Any]] = []
    detailed: Dict[str, Any] = {}

    for target in targets:
        true_col, pred_col = f"true__{target}", f"pred__{target}"
        if true_col not in df.columns or pred_col not in df.columns:
            continue

        sub = df[[true_col, pred_col]].dropna().copy()
        if sub.empty:
            continue

        y_true = sub[true_col].astype(str)
        y_pred = sub[pred_col].astype(str)
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        compact_rows.append(
            {
                "branch": branch,
                "target": target,
                "n_total_rows": len(df),
                "n_scored": len(sub),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            }
        )

        detailed[target] = {
            "labels": labels,
            "classification_report": classification_report(
                y_true, y_pred, labels=labels, output_dict=True, zero_division=0
            ),
        }

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), max(5, len(labels) * 0.6)))
        im = ax.imshow(cm, aspect="auto")
        ax.set_title(f"{branch} – {target}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(paths.plots_dir / f"{branch}_{target}_confusion.png", dpi=180)
        plt.close(fig)

    return compact_rows, detailed


def persist_outputs_so_far(
    prediction_paths: Dict[str, Path],
    active_targets: Sequence[str],
    paths: RuntimePaths,
) -> None:
    merged_preds = []
    compact_rows_all: List[Dict[str, Any]] = []
    detailed_all: Dict[str, Any] = {}
    for branch, pred_path in prediction_paths.items():
        if pred_path.exists():
            merged_preds.append(pd.read_csv(pred_path))
            compact_rows, detailed = evaluate_predictions(pred_path, branch, active_targets, paths)
            compact_rows_all.extend(compact_rows)
            detailed_all[branch] = detailed

    if merged_preds:
        pd.concat(merged_preds, ignore_index=True).to_csv(paths.predictions_dir / "all_branch_predictions.csv", index=False)
    else:
        pd.DataFrame().to_csv(paths.predictions_dir / "all_branch_predictions.csv", index=False)

    pd.DataFrame(compact_rows_all).to_csv(paths.metrics_dir / "metrics_compact.csv", index=False)
    (paths.metrics_dir / "metrics_summary.json").write_text(json.dumps(detailed_all, indent=2), encoding="utf-8")

# -----------------------------------------------------------------------------
# Single-image diagnosis helper
# -----------------------------------------------------------------------------


def single_image_diagnosis(
    args: argparse.Namespace,
    image_path: Path,
    optional_text: Optional[str],
    image_index: ImageIndex,
    vocab: Dict[str, List[str]],
    train_targets: Sequence[str],
    paths: RuntimePaths,
) -> Dict[str, Any]:
    targets = [target for target in CONFIG.single_image_targets if target in train_targets]
    if not targets:
        raise RuntimeError("Requested single-image targets are not available in training vocabulary.")

    qvec = image_index.encode_paths([str(image_path)])[0]
    neighbors = image_index.search_by_vector(qvec, top_k=CONFIG.top_k_images)
    row = pd.Series({"image_path": str(image_path), "patient_text": optional_text, "isic_id": image_path.stem})
    schema = build_response_schema(vocab, targets, include_extras=False)
    raw = call_ollama_json(
        args=args,
        model_name=args.ollama_vlm_model,
        prompt=build_vlm_prompt(row, neighbors, vocab, targets),
        schema=schema,
        branch="single_image_vlm",
        image_id=image_path.stem,
        cache_dir=paths.cache_dir,
        image_path=str(image_path),
    )
    return sanitize_prediction_dict(raw, vocab, targets)


# -----------------------------------------------------------------------------
# Self-test dataset generation
# -----------------------------------------------------------------------------


def create_self_test_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "ISIC-images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    diagnoses = ["nevus", "melanoma"]
    sexes = ["male", "female"]
    sites = ["back", "arm"]
    colors = {
        "nevus": (30, 140, 255),
        "melanoma": (220, 40, 40),
    }

    for i in range(12):
        diag = diagnoses[i % 2]
        sex = sexes[i % 2]
        site = sites[i % 2]
        isic_id = f"ISIC_SELFTEST_{i:04d}"
        img = Image.new("RGB", (96, 96), color=colors[diag])
        img.save(images_dir / f"{isic_id}.png")
        rows.append(
            {
                "isic_id": isic_id,
                "diagnosis_1": diag,
                "diagnosis_2": diag,
                "sex": sex,
                "anatom_site_general": site,
                "age_approx": str(30 + i),
                "melanocytic": "true" if diag == "nevus" else "false",
            }
        )

    pd.DataFrame(rows).to_csv(data_dir / "bcn20000_metadata_2026-01-22.csv", index=False)
    corpus = textwrap.dedent(
        """
        Nevus lesions in this toy dataset are associated with blue-toned images and often marked melanocytic true.
        Melanoma lesions in this toy dataset are associated with red-toned images and often marked melanocytic false.
        This self-test corpus exists only for validating the local pipeline wiring.
        """
    ).strip()
    (data_dir / "skin_lesion_rag_corpus_1000_lines.txt").write_text(corpus, encoding="utf-8")


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def validate_branches(args: argparse.Namespace) -> List[str]:
    branches = list(args.branches)
    if args.disable_local_llm:
        branches = [b for b in branches if b == "knn"]
        if not branches:
            branches = ["knn"]
    return branches


def requested_local_branches(args: argparse.Namespace) -> List[str]:
    return [b for b in list(args.branches) if b in {"llm_rag", "vlm_rag"} and not args.disable_local_llm]


def assert_requested_local_branches_active(args: argparse.Namespace, requested_before_validation: Sequence[str]) -> None:
    missing = [b for b in requested_before_validation if b not in args.branches]
    if missing and args.fail_on_missing_local_branches and not args.self_test:
        raise RuntimeError(
            "Requested local branches were not activated successfully: "
            f"{missing}. Active branches now: {args.branches}. Diagnostics: "
            f"{json.dumps(_ollama_diag(args), ensure_ascii=False)[:8000]}"
        )


def validate_environment(args: argparse.Namespace) -> None:
    configure_optional_hf_token()
    prepare_ollama_runtime_env(args)

    if args.image_backend == "clip" and (not module_available("transformers") or not module_available("torch")):
        LOGGER.warning("CLIP backend dependencies are unavailable. Falling back from --image-backend clip to --image-backend simple.")
        args.image_backend = "simple"

    if args.text_backend == "sentence_transformer" and not module_available("sentence_transformers"):
        LOGGER.warning("sentence-transformers is not installed. Falling back from --text-backend sentence_transformer to --text-backend simple.")
        args.text_backend = "simple"

    requested_local = requested_local_branches(args)
    LOGGER.info("Requested branches at validation: %s", args.branches)
    if not requested_local:
        return

    if not ollama_is_reachable(args, timeout=3):
        maybe_install_ollama(args)
        started = maybe_start_ollama_server(args)
        if started:
            LOGGER.info("Local Ollama server is reachable at %s", args.ollama_base_url)

    if not ollama_is_reachable(args, timeout=5):
        diag = _ollama_diag(args)
        msg = (
            f"Ollama is not reachable at {args.ollama_base_url}. "
            "The pipeline attempted local installation and local server startup before inference. "
            f"Preflight diagnostics: {json.dumps(diag, ensure_ascii=False)}"
        )
        raise RuntimeError(msg)

    model_names, missing_by_branch = ensure_requested_ollama_models(args)
    if missing_by_branch:
        diag = _ollama_diag(args)
        raise RuntimeError(
            "Requested Ollama branches could not be activated because required models are missing: "
            f"{missing_by_branch}. Diagnostics: {json.dumps(diag, ensure_ascii=False)}"
        )

    warmup_failures: Dict[str, str] = {}
    if "llm_rag" in requested_local:
        try:
            warmup_ollama_model(args, args.ollama_llm_model, vision=False)
            LOGGER.info("Warmup succeeded for Ollama text model: %s", args.ollama_llm_model)
        except Exception as exc:
            warmup_failures["llm_rag"] = str(exc)
    if "vlm_rag" in requested_local:
        try:
            warmup_ollama_model(args, args.ollama_vlm_model, vision=True)
            LOGGER.info("Warmup succeeded for Ollama vision model: %s", args.ollama_vlm_model)
        except Exception as exc:
            warmup_failures["vlm_rag"] = str(exc)

    if warmup_failures:
        diag = _ollama_diag(args)
        raise RuntimeError(
            "Requested Ollama branches failed during model warmup: "
            f"{warmup_failures}. Diagnostics: {json.dumps(diag, ensure_ascii=False)}"
        )

    keep_branches: List[str] = []
    for branch in args.branches:
        if branch == "knn":
            keep_branches.append(branch)
        elif branch in requested_local:
            keep_branches.append(branch)

    args.branches = keep_branches or ["knn"]
    args.disable_local_llm = not any(b in {"llm_rag", "vlm_rag"} for b in args.branches)

    if requested_local:
        assert_requested_local_branches_active(args, requested_local)
        active_local = {
            branch: (args.ollama_llm_model if branch == "llm_rag" else args.ollama_vlm_model)
            for branch in args.branches
            if branch in {"llm_rag", "vlm_rag"}
        }
        LOGGER.info("Active local Ollama branches after validation: %s", active_local)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    if args.self_test:
        create_self_test_dataset(args.data_dir)
        args.disable_local_llm = True
        args.branches = ["knn"]
        if args.image_backend == "clip":
            args.image_backend = "simple"
        if args.text_backend == "sentence_transformer":
            args.text_backend = "simple"

    CONFIG.ollama_base_url = args.ollama_base_url
    CONFIG.ollama_llm_model = args.ollama_llm_model
    CONFIG.ollama_vlm_model = args.ollama_vlm_model
    requested_before_validation = list(args.branches)
    args.branches = validate_branches(args)
    LOGGER.info("CLI requested branches before validation: %s", requested_before_validation)
    validate_environment(args)
    preferred_branch_order = ["knn", "vlm_rag", "llm_rag"]
    reordered = [b for b in preferred_branch_order if b in args.branches]
    reordered.extend([b for b in args.branches if b not in reordered])
    args.branches = reordered
    assert_requested_local_branches_active(args, [b for b in requested_before_validation if b in {"llm_rag", "vlm_rag"} and not args.disable_local_llm])
    LOGGER.info("Branches that will actually run: %s", args.branches)

    runtime = runtime_info()
    print_runtime_summary(runtime)
    paths = build_paths(args.data_dir, args.output_dir)

    require_exists(paths.data_dir, "data directory")
    metadata_path = find_metadata_file(paths.data_dir)
    images_dir = image_root(paths.data_dir)
    text_path = find_text_file(paths.data_dir)

    maybe_invalidate_on_manifest_change(metadata_path, images_dir, text_path, args, paths)

    LOGGER.info("Metadata file: %s", metadata_path)
    LOGGER.info("Images folder:  %s", images_dir)
    LOGGER.info("Text file:      %s", text_path)

    metadata = prepare_metadata(read_table(metadata_path))
    image_paths = collect_image_paths(images_dir)
    image_map, duplicate_stems = build_image_map(image_paths)
    merged = attach_image_paths(metadata, image_map)

    full_metadata_matched_df = merged.copy()
    full_df = merged[merged["image_path"].notna()].copy().reset_index(drop=True)

    full_metadata_matched_df.to_csv(paths.artifacts_dir / "full_metadata_matched_df.csv", index=False)
    full_df.to_csv(paths.artifacts_dir / "full_df.csv", index=False)

    LOGGER.info("=" * 80)
    LOGGER.info("Ingestion summary")
    LOGGER.info("=" * 80)
    LOGGER.info("Total metadata rows: %s", len(metadata))
    LOGGER.info("Images found: %s", len(image_paths))
    LOGGER.info("Matched rows across full metadata: %s", len(full_df))
    LOGGER.info("Unmatched metadata rows: %s", len(metadata) - len(full_df))
    LOGGER.info("Duplicate image stems: %s", len(duplicate_stems))
    LOGGER.info("=" * 80)

    if len(full_df) < 10:
        raise RuntimeError("Too few matched rows. Check that image filenames match metadata isic_id values.")

    train_df_original, test_df = split_dataset(full_df)
    aug_cfg = augmentation_config_dict()
    aug_dir = paths.artifacts_dir / aug_cfg["output_subdir"]
    if aug_dir.exists():
        shutil.rmtree(aug_dir)
    aug_dir.mkdir(parents=True, exist_ok=True)

    if aug_cfg["enabled"]:
        train_df = create_augmented_train_df(
            train_df=train_df_original,
            output_dir=aug_dir,
            copies_per_image=int(aug_cfg["copies_per_image"]),
            cfg=aug_cfg,
        )
    else:
        train_df = train_df_original.copy()
        train_df["is_augmented"] = False
        train_df["source_isic_id"] = train_df["isic_id"]
        train_df["augmentation_copy_idx"] = 0
        train_df["augmentation_seed"] = pd.NA

    train_df_original.to_csv(paths.artifacts_dir / "train_df_original.csv", index=False)
    train_df.to_csv(paths.artifacts_dir / "train_df.csv", index=False)
    train_df.to_csv(paths.artifacts_dir / "train_df_augmented.csv", index=False)
    test_df.to_csv(paths.artifacts_dir / "test_df.csv", index=False)

    active_targets = infer_active_targets(train_df_original, test_df)
    if not active_targets:
        raise RuntimeError("No active targets were found in both train and test splits.")

    vocab = build_label_vocab_train_only(train_df_original, active_targets)
    (paths.artifacts_dir / "label_vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    LOGGER.info("Original train rows: %s", len(train_df_original))
    LOGGER.info("Augmented train rows: %s", len(train_df))
    LOGGER.info("Test rows: %s", len(test_df))
    LOGGER.info("Active targets: %s", active_targets)

    needs_text_index = "llm_rag" in args.branches
    text_index = None
    chunk_df = pd.DataFrame(columns=["chunk_id", "source_id", "chunk_text"])
    if needs_text_index:
        text_sources = load_text_sources(text_path)
        chunk_df = chunk_text_sources(text_sources)
        chunk_df.to_csv(paths.artifacts_dir / "chunk_df.csv", index=False)
        if not chunk_df.empty:
            try:
                text_index = TextChunkIndex(args.text_backend, runtime)
                text_index.build(chunk_df, embedding_cache_path=paths.artifacts_dir / f"text_chunk_embeddings_{args.text_backend}.npy")
                LOGGER.info("Text chunks indexed: %s", len(chunk_df))
            except Exception as exc:
                if args.text_backend != "simple":
                    LOGGER.warning("Text backend '%s' failed (%s). Falling back to simple text backend.", args.text_backend, exc)
                    args.text_backend = "simple"
                    text_index = TextChunkIndex(args.text_backend, runtime)
                    text_index.build(chunk_df, embedding_cache_path=paths.artifacts_dir / f"text_chunk_embeddings_{args.text_backend}.npy")
                    LOGGER.info("Text chunks indexed with fallback simple backend: %s", len(chunk_df))
                else:
                    raise
        else:
            LOGGER.info("No text corpus found or no text chunks created.")
    else:
        LOGGER.info("Skipping text index because llm_rag is not an active branch.")

    image_index = None
    image_backend_used = args.image_backend
    try:
        image_index = ImageIndex(args.image_backend, runtime)
        image_index.build(train_df, embedding_cache_path=paths.artifacts_dir / f"train_image_embeddings_{args.image_backend}.npy")
        image_index.save(paths.artifacts_dir / f"train_image_index_{args.image_backend}")
    except Exception as exc:
        if args.image_backend != "simple":
            LOGGER.warning("Image backend '%s' failed (%s). Falling back to simple image backend.", args.image_backend, exc)
            args.image_backend = "simple"
            image_backend_used = args.image_backend
            image_index = ImageIndex(args.image_backend, runtime)
            image_index.build(train_df, embedding_cache_path=paths.artifacts_dir / f"train_image_embeddings_{args.image_backend}.npy")
            image_index.save(paths.artifacts_dir / f"train_image_index_{args.image_backend}")
        else:
            raise
    clean_cuda()

    test_image_embeddings_path = paths.artifacts_dir / f"test_image_embeddings_{image_backend_used}.npy"
    if test_image_embeddings_path.exists():
        test_embeddings = np.load(test_image_embeddings_path)
    else:
        test_embeddings = image_index.encode_paths(test_df["image_path"].astype(str).tolist())
        np.save(test_image_embeddings_path, test_embeddings)
    clean_cuda()

    neighbor_cache_path = paths.artifacts_dir / f"test_neighbor_cache_{image_backend_used}.pkl"
    if neighbor_cache_path.exists():
        neighbor_cache = load_pickle(neighbor_cache_path)
    else:
        neighbor_cache = image_index.search_many_by_vectors(test_embeddings, top_k=CONFIG.top_k_images)
        save_pickle(neighbor_cache, neighbor_cache_path)

    prediction_paths: Dict[str, Path] = {}
    vlm_context_map: Dict[str, Dict[str, Any]] = {}
    for branch in args.branches:
        out_path, err_path = run_branch_predictions(
            branch=branch,
            test_df=test_df,
            targets=active_targets,
            vocab=vocab,
            image_index=image_index,
            test_embeddings=test_embeddings,
            text_index=text_index if branch == "llm_rag" else None,
            args=args,
            paths=paths,
            neighbor_cache=neighbor_cache,
            prior_context_map=vlm_context_map if branch == "llm_rag" else None,
        )
        prediction_paths[branch] = out_path
        pred_count = len(pd.read_csv(out_path)) if out_path.exists() else 0
        err_count = len(pd.read_csv(err_path)) if err_path.exists() else 0
        LOGGER.info("%s: predictions=%s, errors=%s, expected_test_rows=%s", branch, pred_count, err_count, len(test_df))
        if branch == "vlm_rag":
            vlm_context_map = load_branch_prediction_map(out_path, active_targets)
            LOGGER.info("Loaded VLM context map for %s samples", len(vlm_context_map))
        persist_outputs_so_far(prediction_paths, active_targets, paths)
        LOGGER.info("Saved cumulative outputs after branch: %s", branch)

    persist_outputs_so_far(prediction_paths, active_targets, paths)

    LOGGER.info("=" * 80)
    LOGGER.info("Pipeline complete.")
    LOGGER.info("Predictions: %s", paths.predictions_dir)
    LOGGER.info("Metrics:     %s", paths.metrics_dir)
    LOGGER.info("Plots:       %s", paths.plots_dir)
    LOGGER.info("=" * 80)

    if not args.skip_single_image_demo and image_paths and not args.disable_local_llm and any(branch in {"llm_rag", "vlm_rag"} for branch in args.branches):
        demo_path = Path(image_paths[0])
        LOGGER.info("Example single-image diagnosis on: %s", demo_path.name)
        try:
            single_pred = single_image_diagnosis(
                args=args,
                image_path=demo_path,
                optional_text=None,
                image_index=image_index,
                vocab=vocab,
                train_targets=active_targets,
                paths=paths,
            )
            LOGGER.info("Single-image prediction: %s", json.dumps(single_pred, indent=2))
        except Exception as exc:
            LOGGER.warning("Single-image demo failed but the pipeline outputs were already saved: %s", exc)



# -----------------------------------------------------------------------------
# Embedded prerequisite stage
# -----------------------------------------------------------------------------

_EMBEDDED_PREREQUISITE_SOURCE = 'from __future__ import annotations\n\n"""\nPrepare a local data directory beside this script for the dermoscopy pipeline.\n\nPriority order:\n1. If <script_dir>/data already contains the needed pieces, reuse them.\n2. If <script_dir>/sop or nearby sop/ exists, copy only the missing pieces into <script_dir>/data.\n3. If custom URLs are supplied through environment variables, download only the missing pieces.\n4. Otherwise try the public ISIC BCN20000 collection defaults.\n\nData is always placed in a data/ folder next to this file.\n"""\n\nimport os\nimport shutil\nimport tarfile\nimport tempfile\nimport time\nimport json\nimport subprocess\nimport sys\nimport platform\nimport stat\nimport urllib.parse\nimport urllib.request\nimport zipfile\nfrom html.parser import HTMLParser\nfrom pathlib import Path\nfrom typing import Iterable, Optional\n\nSCRIPT_DIR = Path(__file__).resolve().parent\nDATA_DIR = SCRIPT_DIR / "data"\nIMAGE_DIR = DATA_DIR / "ISIC-images"\nMETADATA_STEM = "bcn20000_metadata_2026-01-22"\nCORPUS_STEM = "skin_lesion_rag_corpus_1000_lines"\nVALID_METADATA_EXTS = [".xlsx", ".xls", ".csv"]\nVALID_CORPUS_EXTS = [".txt", ".md", ".csv", ".xlsx", ".xls"]\nVALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}\n\nOLLAMA_INSTALL_DIR = SCRIPT_DIR / ".local_ollama"\nOLLAMA_MODELS_DIR = Path(os.getenv("OLLAMA_MODELS", str(Path.home() / "ollama_models"))).expanduser()\nOLLAMA_TMP_DIR = SCRIPT_DIR / ".ollama_tmp"\n_OLLAMA_HOST_ENV = os.getenv("OLLAMA_HOST", "").strip()\n_OLLAMA_BASE_ENV = os.getenv("OLLAMA_BASE_URL", "").strip()\nif _OLLAMA_BASE_ENV:\n    OLLAMA_BASE_URL = _OLLAMA_BASE_ENV\nelif _OLLAMA_HOST_ENV:\n    OLLAMA_BASE_URL = _OLLAMA_HOST_ENV if _OLLAMA_HOST_ENV.startswith(("http://", "https://")) else f"http://{_OLLAMA_HOST_ENV}"\nelse:\n    OLLAMA_BASE_URL = "http://127.0.0.1:11434"\nOLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")\nOLLAMA_VLM_MODEL = os.getenv("OLLAMA_VLM_MODEL", "llama3.2-vision:11b")\nPREPARE_OLLAMA = os.getenv("DERM_PREPARE_OLLAMA", "1") != "0"\nAUTO_PULL_OLLAMA_MODELS = os.getenv("DERM_AUTO_PULL_OLLAMA_MODELS", "1") != "0"\n\nDEFAULT_COLLECTION_PAGE_URL = (\n    "https://api.isic-archive.com/collections/249/"\n    "?cursor=cD0yMDE5LTAzLTIyKzIwJTNBMjklM0EwNy45ODcwMDAlMkIwMCUzQTAw"\n)\nDEFAULT_METADATA_URL = "https://api.isic-archive.com/collections/249/metadata/"\nDEFAULT_COLLECTION_ARCHIVE_CANDIDATES = [\n    "https://api.isic-archive.com/collections/249/download/",\n    "https://api.isic-archive.com/collections/249/download",\n    "https://api.isic-archive.com/collections/249/archive/",\n    "https://api.isic-archive.com/collections/249/archive",\n    "https://api.isic-archive.com/collections/249/?download=1",\n]\nDEFAULT_HEADERS = {\n    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",\n    "Accept": "*/*",\n}\n\n\nclass LinkParser(HTMLParser):\n    def __init__(self) -> None:\n        super().__init__()\n        self.links: list[str] = []\n\n    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:\n        if tag.lower() != "a":\n            return\n        for k, v in attrs:\n            if k.lower() == "href" and v:\n                self.links.append(v)\n\n\ndef ensure_data_dir() -> None:\n    DATA_DIR.mkdir(parents=True, exist_ok=True)\n\n\ndef find_existing_file(root: Path, stem: str, exts: list[str]) -> Optional[Path]:\n    if not root.exists():\n        return None\n    for ext in exts:\n        candidate = root / f"{stem}{ext}"\n        if candidate.exists():\n            return candidate\n    for item in root.iterdir():\n        if item.is_file() and item.stem == stem:\n            return item\n    return None\n\n\ndef count_images(root: Path) -> int:\n    if not root.exists():\n        return 0\n    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS)\n\n\ndef metadata_ready() -> bool:\n    return find_existing_file(DATA_DIR, METADATA_STEM, VALID_METADATA_EXTS) is not None\n\n\ndef images_ready() -> bool:\n    return count_images(IMAGE_DIR) > 0\n\n\ndef data_ready() -> bool:\n    return metadata_ready() and images_ready()\n\n\ndef validate_prepared_data() -> None:\n    ensure_data_dir()\n    metadata = find_existing_file(DATA_DIR, METADATA_STEM, VALID_METADATA_EXTS)\n    image_count = count_images(IMAGE_DIR)\n    if metadata is None:\n        raise RuntimeError(f"Metadata file not found in {DATA_DIR}")\n    if image_count <= 0:\n        raise RuntimeError(f"No images found in {IMAGE_DIR}")\n    print(f"Validation passed: metadata={metadata.name}, images={image_count}, data_dir={DATA_DIR}")\n\n\ndef copy_if_present(src: Optional[Path], dst: Path) -> None:\n    if src is None:\n        return\n    dst.parent.mkdir(parents=True, exist_ok=True)\n    shutil.copy2(src, dst)\n\n\ndef copy_tree_if_present(src: Path, dst: Path, overwrite: bool = False) -> None:\n    if not src.exists():\n        return\n    if dst.exists() and overwrite:\n        shutil.rmtree(dst)\n    if dst.exists() and not overwrite:\n        return\n    shutil.copytree(src, dst)\n\n\ndef try_copy_from_local_sop() -> bool:\n    candidates = [SCRIPT_DIR / "sop", SCRIPT_DIR.parent / "sop", Path.cwd() / "sop", Path.cwd().parent / "sop"]\n    changed = False\n    for sop_dir in candidates:\n        if not sop_dir.exists():\n            continue\n        metadata = find_existing_file(sop_dir, METADATA_STEM, VALID_METADATA_EXTS)\n        corpus = find_existing_file(sop_dir, CORPUS_STEM, VALID_CORPUS_EXTS)\n        image_src = sop_dir / "ISIC-images"\n        ensure_data_dir()\n        if metadata is not None and not metadata_ready():\n            copy_if_present(metadata, DATA_DIR / metadata.name)\n            changed = True\n        if corpus is not None and find_existing_file(DATA_DIR, CORPUS_STEM, VALID_CORPUS_EXTS) is None:\n            copy_if_present(corpus, DATA_DIR / corpus.name)\n            changed = True\n        if count_images(image_src) > 0 and not images_ready():\n            copy_tree_if_present(image_src, IMAGE_DIR, overwrite=True)\n            changed = True\n        if changed or data_ready():\n            print(f"Used local source: {sop_dir.resolve()}")\n            return data_ready()\n    return data_ready()\n\n\ndef build_request(url: str):\n    return urllib.request.Request(url, headers=DEFAULT_HEADERS)\n\n\ndef get_url_bytes(url: str, timeout: int = 120) -> bytes:\n    with urllib.request.urlopen(build_request(url), timeout=timeout) as resp:\n        return resp.read()\n\n\ndef download_file(url: str, dst: Path, timeout: int = 300) -> None:\n    dst.parent.mkdir(parents=True, exist_ok=True)\n    print(f"Downloading {url} -> {dst}")\n    with urllib.request.urlopen(build_request(url), timeout=timeout) as resp, open(dst, "wb") as f:\n        shutil.copyfileobj(resp, f)\n\n\ndef download_to_temp(url: str, suffix: str = "") -> Path:\n    fd, tmp_name = tempfile.mkstemp(suffix=suffix)\n    os.close(fd)\n    tmp_path = Path(tmp_name)\n    try:\n        download_file(url, tmp_path)\n        return tmp_path\n    except Exception:\n        if tmp_path.exists():\n            tmp_path.unlink()\n        raise\n\n\ndef unpack_archive(archive_path: Path, target_dir: Path) -> None:\n    target_dir.mkdir(parents=True, exist_ok=True)\n    suffixes = archive_path.suffixes\n    if archive_path.suffix.lower() == ".zip" or suffixes[-1:] == [".zip"]:\n        with zipfile.ZipFile(archive_path, "r") as zf:\n            zf.extractall(target_dir)\n        return\n    if suffixes[-2:] in ([".tar", ".gz"], [".tar", ".bz2"], [".tar", ".xz"]):\n        with tarfile.open(archive_path, "r:*") as tf:\n            tf.extractall(target_dir)\n        return\n    raise ValueError(f"Unsupported archive type: {archive_path}")\n\n\ndef normalize_extracted_images(search_root: Path) -> bool:\n    if count_images(IMAGE_DIR) > 0:\n        return True\n    nested = [p for p in search_root.rglob("ISIC-images") if p.is_dir()]\n    if nested:\n        src = nested[0]\n        if IMAGE_DIR.exists():\n            shutil.rmtree(IMAGE_DIR)\n        shutil.move(str(src), str(IMAGE_DIR))\n        return count_images(IMAGE_DIR) > 0\n    image_files = [p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]\n    if image_files:\n        IMAGE_DIR.mkdir(parents=True, exist_ok=True)\n        for p in image_files:\n            target = IMAGE_DIR / p.name\n            if not target.exists():\n                shutil.move(str(p), str(target))\n        return count_images(IMAGE_DIR) > 0\n    return False\n\n\ndef parse_links(html_text: str, base_url: str) -> list[str]:\n    parser = LinkParser()\n    parser.feed(html_text)\n    return [urllib.parse.urljoin(base_url, href) for href in parser.links]\n\n\ndef infer_metadata_name_from_url(url: str) -> str:\n    path_name = Path(urllib.parse.urlparse(url).path).name\n    return path_name or f"{METADATA_STEM}.csv"\n\n\ndef fetch_collection_page_links(page_url: str) -> list[str]:\n    try:\n        html_text = get_url_bytes(page_url).decode("utf-8", errors="ignore")\n    except Exception:\n        return []\n    return list(dict.fromkeys(parse_links(html_text, page_url)))\n\n\ndef candidate_archive_urls(collection_page_url: str) -> list[str]:\n    parsed = urllib.parse.urlparse(collection_page_url)\n    base = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))\n    if not base.endswith("/"):\n        base += "/"\n    candidates = list(DEFAULT_COLLECTION_ARCHIVE_CANDIDATES)\n    candidates.extend([\n        urllib.parse.urljoin(base, "download/"),\n        urllib.parse.urljoin(base, "download"),\n        urllib.parse.urljoin(base, "archive/"),\n        urllib.parse.urljoin(base, "archive"),\n    ])\n    for link in fetch_collection_page_links(collection_page_url):\n        low = link.lower()\n        if any(key in low for key in ["/download", "download=", ".zip", ".tar", ".tgz"]):\n            candidates.append(link)\n    seen = set()\n    uniq = []\n    for url in candidates:\n        if url not in seen:\n            uniq.append(url)\n            seen.add(url)\n    return uniq\n\n\ndef looks_like_archive(path: Path) -> bool:\n    if zipfile.is_zipfile(path):\n        return True\n    try:\n        with tarfile.open(path, "r:*"):\n            return True\n    except Exception:\n        return False\n\n\ndef try_download_collection_archive(urls: Iterable[str]) -> bool:\n    if images_ready():\n        return True\n    last_error: Optional[Exception] = None\n    for url in urls:\n        tmp_path: Optional[Path] = None\n        try:\n            suffix = Path(urllib.parse.urlparse(url).path).suffix or ".bin"\n            tmp_path = download_to_temp(url, suffix=suffix)\n            if not looks_like_archive(tmp_path):\n                text = tmp_path.read_text(encoding="utf-8", errors="ignore")[:500].lower()\n                if "sign in" in text or "<html" in text:\n                    raise RuntimeError(f"URL returned HTML instead of an archive: {url}")\n                raise RuntimeError(f"URL did not return a supported archive: {url}")\n            unpack_archive(tmp_path, DATA_DIR)\n            if not normalize_extracted_images(DATA_DIR):\n                raise RuntimeError(f"Archive downloaded from {url} did not contain usable images.")\n            print(f"Downloaded collection archive from: {url}")\n            return True\n        except Exception as exc:\n            last_error = exc\n        finally:\n            if tmp_path is not None and tmp_path.exists():\n                tmp_path.unlink()\n    if last_error is not None:\n        print(f"Collection archive download attempts failed: {last_error}")\n    return images_ready()\n\n\ndef try_download_metadata(url: str) -> bool:\n    if metadata_ready():\n        return True\n    try:\n        ensure_data_dir()\n        name = infer_metadata_name_from_url(url)\n        if "." not in name:\n            name = f"{METADATA_STEM}.csv"\n        dst = DATA_DIR / name\n        download_file(url, dst)\n        normalized = DATA_DIR / f"{METADATA_STEM}{dst.suffix.lower() or \'.csv\'}"\n        if dst != normalized:\n            if normalized.exists():\n                normalized.unlink()\n            dst.rename(normalized)\n        return metadata_ready()\n    except Exception as exc:\n        print(f"Metadata download failed from {url}: {exc}")\n        return metadata_ready()\n\n\ndef try_download_corpus(corpus_url: Optional[str]) -> bool:\n    if not corpus_url:\n        return False\n    if find_existing_file(DATA_DIR, CORPUS_STEM, VALID_CORPUS_EXTS) is not None:\n        return True\n    try:\n        name = Path(urllib.parse.urlparse(corpus_url).path).name or f"{CORPUS_STEM}.txt"\n        download_file(corpus_url, DATA_DIR / name)\n        return True\n    except Exception as exc:\n        print(f"Optional corpus download failed from {corpus_url}: {exc}")\n        return False\n\n\ndef try_download_from_isic_defaults() -> bool:\n    collection_page = os.getenv("ISIC_COLLECTION_PAGE_URL", DEFAULT_COLLECTION_PAGE_URL)\n    metadata_url = os.getenv("ISIC_METADATA_URL", DEFAULT_METADATA_URL)\n    archive_urls = candidate_archive_urls(collection_page)\n    metadata_ok = try_download_metadata(metadata_url)\n    images_ok = try_download_collection_archive(archive_urls)\n    try_download_corpus(os.getenv("CORPUS_URL"))\n    return metadata_ok and images_ok and data_ready()\n\n\ndef try_download_from_env() -> bool:\n    metadata_url = os.getenv("METADATA_URL") or os.getenv("ISIC_METADATA_URL")\n    images_url = os.getenv("IMAGES_URL") or os.getenv("ISIC_IMAGES_URL")\n    corpus_url = os.getenv("CORPUS_URL")\n    if not metadata_url and not images_url and not corpus_url:\n        return False\n    metadata_ok = try_download_metadata(metadata_url) if metadata_url else metadata_ready()\n    images_ok = try_download_collection_archive([images_url]) if images_url else images_ready()\n    try_download_corpus(corpus_url)\n    return metadata_ok and images_ok and data_ready()\n\n\ndef _detect_ollama_download_url() -> str:\n    machine = platform.machine().lower()\n    if machine in {"x86_64", "amd64"}:\n        arch = "amd64"\n    elif machine in {"aarch64", "arm64"}:\n        arch = "arm64"\n    else:\n        raise RuntimeError(f"Unsupported architecture for automatic Ollama installation: {machine}")\n    return f"https://ollama.com/download/ollama-linux-{arch}.tar.zst"\n\n\ndef _candidate_ollama_bins() -> list[Path]:\n    cands = [OLLAMA_INSTALL_DIR / "bin" / "ollama", OLLAMA_INSTALL_DIR / "usr" / "bin" / "ollama"]\n    path_bin = shutil.which("ollama")\n    if path_bin:\n        cands.append(Path(path_bin))\n    out: list[Path] = []\n    seen = set()\n    for p in cands:\n        key = str(p)\n        if key not in seen:\n            seen.add(key)\n            out.append(p)\n    return out\n\n\ndef _prepend_env_path(name: str, value: Path) -> None:\n    current = os.environ.get(name, "")\n    parts = [p for p in current.split(os.pathsep) if p]\n    value = str(value.resolve())\n    if value not in parts:\n        os.environ[name] = os.pathsep.join([value] + parts) if parts else value\n\n\ndef prepare_ollama_env() -> None:\n    OLLAMA_INSTALL_DIR.mkdir(parents=True, exist_ok=True)\n    OLLAMA_MODELS_DIR.mkdir(parents=True, exist_ok=True)\n    OLLAMA_TMP_DIR.mkdir(parents=True, exist_ok=True)\n    if (OLLAMA_INSTALL_DIR / "bin").exists():\n        _prepend_env_path("PATH", OLLAMA_INSTALL_DIR / "bin")\n    if (OLLAMA_INSTALL_DIR / "lib" / "ollama").exists():\n        _prepend_env_path("LD_LIBRARY_PATH", OLLAMA_INSTALL_DIR / "lib" / "ollama")\n    if (OLLAMA_INSTALL_DIR / "lib").exists():\n        _prepend_env_path("LD_LIBRARY_PATH", OLLAMA_INSTALL_DIR / "lib")\n    os.environ.setdefault("OLLAMA_MODELS", str(OLLAMA_MODELS_DIR.resolve()))\n    os.environ.setdefault("OLLAMA_TMPDIR", str(OLLAMA_TMP_DIR.resolve()))\n    os.environ.setdefault("GODEBUG", "http2client=0")\n    parsed = urllib.parse.urlparse(OLLAMA_BASE_URL)\n    os.environ.setdefault("OLLAMA_HOST", f"{parsed.hostname or \'127.0.0.1\'}:{parsed.port or 11434}")\n\n\ndef _ensure_zstandard():\n    try:\n        import zstandard as zstd\n        return zstd\n    except Exception:\n        subprocess.run([sys.executable, "-m", "pip", "install", "zstandard"], check=True)\n        import zstandard as zstd\n        return zstd\n\n\ndef _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:\n    dest = dest.resolve()\n    for member in tf.getmembers():\n        target = (dest / member.name).resolve()\n        if not str(target).startswith(str(dest)):\n            raise RuntimeError(f"Unsafe tar member during Ollama installation: {member.name}")\n    tf.extractall(dest)\n\n\ndef install_local_ollama_if_needed() -> Path:\n    prepare_ollama_env()\n    for cand in _candidate_ollama_bins():\n        if cand.exists() and os.access(cand, os.X_OK):\n            return cand.resolve()\n    url = os.getenv("OLLAMA_INSTALL_URL", "").strip() or _detect_ollama_download_url()\n    print(f"Installing local Ollama runtime into {OLLAMA_INSTALL_DIR} from {url}")\n    tmp_root = Path(tempfile.mkdtemp(prefix="ollama_setup_"))\n    archive_zst = tmp_root / "ollama.tar.zst"\n    archive_tar = tmp_root / "ollama.tar"\n    try:\n        urllib.request.urlretrieve(url, archive_zst)\n        zstd = _ensure_zstandard()\n        with open(archive_zst, "rb") as src_f, open(archive_tar, "wb") as dst_f:\n            zstd.ZstdDecompressor().copy_stream(src_f, dst_f)\n        with tarfile.open(archive_tar, "r:") as tf:\n            _safe_extract_tar(tf, OLLAMA_INSTALL_DIR)\n        for cand in _candidate_ollama_bins():\n            if cand.exists():\n                try:\n                    cand.chmod(cand.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)\n                except Exception:\n                    pass\n                prepare_ollama_env()\n                return cand.resolve()\n        raise RuntimeError(f"Installed Ollama but could not find the binary under {OLLAMA_INSTALL_DIR}")\n    finally:\n        shutil.rmtree(tmp_root, ignore_errors=True)\n\n\ndef ollama_http_ok(path: str = "/api/tags", timeout: float = 3.0) -> bool:\n    try:\n        with urllib.request.urlopen(OLLAMA_BASE_URL.rstrip("/") + path, timeout=timeout) as resp:\n            return 200 <= resp.status < 300\n    except Exception:\n        return False\n\n\ndef start_ollama_server_if_needed() -> tuple[Optional[subprocess.Popen], Path]:\n    prepare_ollama_env()\n    log_path = SCRIPT_DIR / "ollama_setup.log"\n    if ollama_http_ok():\n        return None, log_path\n    ollama_bin = install_local_ollama_if_needed()\n    log_f = open(log_path, "a", encoding="utf-8")\n    proc = subprocess.Popen([str(ollama_bin), "serve"], stdout=log_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env=os.environ.copy())\n    deadline = time.time() + 180\n    while time.time() < deadline:\n        if proc.poll() is not None:\n            raise RuntimeError(f"ollama serve exited early with code {proc.returncode}. Check {log_path}")\n        if ollama_http_ok():\n            return proc, log_path\n        time.sleep(1.5)\n    raise RuntimeError(f"Timed out waiting for Ollama to become healthy. Check {log_path}")\n\n\ndef run_ollama_cmd(*args: str, timeout: int = 7200) -> subprocess.CompletedProcess:\n    ollama_bin = install_local_ollama_if_needed()\n    env = os.environ.copy()\n    env.setdefault("GODEBUG", "http2client=0")\n    return subprocess.run([str(ollama_bin), *args], capture_output=True, text=True, timeout=timeout, env=env)\n\n\ndef registry_diag() -> dict:\n    out = {}\n    for url in ["https://registry.ollama.ai/v2/", "https://ollama.com/download/"]:\n        try:\n            req = urllib.request.Request(url, headers=DEFAULT_HEADERS)\n            with urllib.request.urlopen(req, timeout=20) as resp:\n                out[url] = {"ok": True, "status": getattr(resp, "status", None)}\n        except Exception as exc:\n            out[url] = {"ok": False, "error": str(exc)}\n    return out\n\n\ndef pull_ollama_model_with_retries(model: str, retries: int = 5) -> None:\n    last_err = None\n    for attempt in range(1, max(1, retries) + 1):\n        print(f"Pulling Ollama model: {model} (attempt {attempt}/{retries})")\n        cp = run_ollama_cmd("pull", model, timeout=14400)\n        if cp.returncode == 0:\n            return\n        last_err = f"stdout={cp.stdout[-2000:]} stderr={cp.stderr[-2000:]}"\n        print(f"Attempt {attempt} failed for {model}: {last_err}")\n        if attempt < retries:\n            time.sleep(min(60, 5 * attempt))\n    raise RuntimeError(f"Failed to pull {model} after {retries} attempts. Registry diagnostics: {json.dumps(registry_diag())}. Last error: {last_err}")\n\n\ndef ensure_ollama_models_ready() -> None:\n    if not PREPARE_OLLAMA:\n        print("Skipping Ollama preparation because DERM_PREPARE_OLLAMA=0")\n        return\n    proc = None\n    try:\n        proc, log_path = start_ollama_server_if_needed()\n        cp = run_ollama_cmd("list", timeout=60)\n        names = set()\n        if cp.returncode == 0:\n            for line in cp.stdout.splitlines()[1:]:\n                parts = line.split()\n                if parts:\n                    names.add(parts[0])\n        required = [OLLAMA_LLM_MODEL, OLLAMA_VLM_MODEL]\n        missing = [m for m in required if m not in names]\n        if missing and not AUTO_PULL_OLLAMA_MODELS:\n            raise RuntimeError(f"Missing Ollama models and auto-pull disabled: {missing}")\n        retries = max(1, int(os.getenv("DERM_OLLAMA_PULL_RETRIES", "5")))\n        for model in missing:\n            pull_ollama_model_with_retries(model, retries=retries)\n        cp = run_ollama_cmd("list", timeout=60)\n        if cp.returncode != 0:\n            raise RuntimeError(f"ollama list failed after pull: stdout={cp.stdout[-2000:]} stderr={cp.stderr[-2000:]}")\n        print("Ollama models ready.")\n        print(cp.stdout)\n        print(f"Ollama setup log: {log_path}")\n    finally:\n        if proc is not None and proc.poll() is None:\n            proc.terminate()\n            try:\n                proc.wait(timeout=10)\n            except Exception:\n                proc.kill()\n\n\ndef main() -> None:\n    print(f"Script dir: {SCRIPT_DIR}")\n    print(f"Target data dir: {DATA_DIR}")\n    ensure_data_dir()\n    ready = False\n    if data_ready():\n        print("No download needed. Required files already exist in the data directory next to this script.")\n        ready = True\n    elif try_copy_from_local_sop() and data_ready():\n        print("Dataset prepared in the data directory next to this script from local sop folder.")\n        ready = True\n    elif try_download_from_env() and data_ready():\n        print("Dataset prepared in the data directory next to this script from custom URLs.")\n        ready = True\n    elif try_download_from_isic_defaults() and data_ready():\n        print("Dataset prepared in the data directory next to this script from ISIC BCN20000 collection actions.")\n        ready = True\n\n    if not ready:\n        raise SystemExit(\n            "Could not prepare the local data directory. If images are already present, make sure metadata is also present. Otherwise keep a local sop/ folder beside this script, or ensure internet access is available for the ISIC download URLs."\n        )\n\n    validate_prepared_data()\n    ensure_ollama_models_ready()\n\n\nif __name__ == "__main__":\n    main()\n'

def _run_embedded_prerequisites() -> None:
    namespace = {
        "__file__": str(Path(__file__).resolve()),
        "__name__": "__embedded_prerequisites__",
        "__package__": None,
    }
    exec(_EMBEDDED_PREREQUISITE_SOURCE, namespace, namespace)
    prep_main = namespace.get("main")
    if prep_main is None:
        raise RuntimeError("Embedded prerequisite stage is missing a main() entrypoint")
    prep_main()


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------



def _resolve_effective_ollama_base_url(args: argparse.Namespace) -> str:
    default_url = CONFIG.ollama_base_url
    cli_url = getattr(args, "ollama_base_url", default_url)
    env_base = os.environ.get("OLLAMA_BASE_URL", "").strip()
    env_host = os.environ.get("OLLAMA_HOST", "").strip()
    if cli_url != default_url:
        return cli_url
    if env_base:
        return env_base
    if env_host:
        return env_host if env_host.startswith(("http://", "https://")) else f"http://{env_host}"
    return cli_url


def _sync_ollama_runtime_args(args: argparse.Namespace) -> None:
    args.ollama_base_url = _resolve_effective_ollama_base_url(args)


def _export_args_to_prereq_env(args: argparse.Namespace) -> None:
    effective_url = _resolve_effective_ollama_base_url(args)
    os.environ["OLLAMA_BASE_URL"] = effective_url
    parsed = urllib.parse.urlparse(effective_url)
    if parsed.netloc:
        os.environ["OLLAMA_HOST"] = parsed.netloc
    os.environ["OLLAMA_LLM_MODEL"] = args.ollama_llm_model
    os.environ["OLLAMA_VLM_MODEL"] = args.ollama_vlm_model
    requested_local = any(b in {"llm_rag", "vlm_rag"} for b in args.branches) and not args.disable_local_llm
    os.environ["DERM_PREPARE_OLLAMA"] = "1" if requested_local else "0"
    os.environ["DERM_AUTO_PULL_OLLAMA_MODELS"] = "1" if args.auto_pull_ollama_models else "0"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _sync_ollama_runtime_args(args)
    setup_logging(args.verbose)
    _export_args_to_prereq_env(args)
    LOGGER.info("Starting embedded prerequisite stage. Requested branches: %s", args.branches)
    LOGGER.info("Using Ollama base URL: %s", args.ollama_base_url)
    _run_embedded_prerequisites()
    run_pipeline(args)


if __name__ == "__main__":
    main()
