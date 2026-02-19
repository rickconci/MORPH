"""
Load MORPH paths from environment / .env.

Set MORPH_REPO_DIR, MORPH_DATA_ROOT, and optionally MORPH_RESULT_DIR in .env
(or export them). Paths in data/*.csv are relative to MORPH_DATA_ROOT unless absolute.

Example .env:
    MORPH_REPO_DIR=/path/to/MORPH
    MORPH_DATA_ROOT=/path/to/SC_PerturbSeq_datasets
    MORPH_RESULT_DIR=/path/to/ML_OUTPUTS/Morph
"""

from __future__ import annotations

import os
from typing import Optional

def _repo_dir_from_file() -> str:
    """Repo root from this file: morph/config.py -> parent of morph/."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv() -> None:
    """Load .env from repo root so MORPH_* env vars are set."""
    try:
        import dotenv
        repo = _repo_dir_from_file()
        env_path = os.path.join(repo, ".env")
        if os.path.isfile(env_path):
            dotenv.load_dotenv(env_path)
    except ImportError:
        pass


_load_dotenv()


def _find_repo_dir() -> str:
    """Repo root: MORPH_REPO_DIR or directory containing morph/ (parent of morph/)."""
    env = os.environ.get("MORPH_REPO_DIR", "").strip()
    if env:
        return os.path.abspath(env)
    return _repo_dir_from_file()


def get_repo_dir() -> str:
    """Return MORPH repo root (directory containing morph/ and data/)."""
    return _find_repo_dir()


def get_data_root() -> str:
    """
    Return base directory for datasets (e.g. SC_PerturbSeq_datasets).
    Paths in data/scdata_file_path.csv and data/perturb_embed_file_path.csv
    are relative to this unless they are absolute.
    """
    root = os.environ.get("MORPH_DATA_ROOT", "").strip()
    if root:
        return os.path.abspath(root)
    # Fallback: repo/data (if you put datasets inside repo)
    return os.path.join(get_repo_dir(), "data")


def get_result_dir() -> Optional[str]:
    """
    Return base directory for checkpoints/results, or None to use repo/result.
    """
    path = os.environ.get("MORPH_RESULT_DIR", "").strip()
    if path:
        return os.path.abspath(path)
    return None


def resolve_data_path(path: str, data_root: Optional[str] = None) -> str:
    """
    Resolve a path from CSV: if absolute, return as-is; else join with data root.

    Args:
        path: Path from scdata_file_path.csv or perturb_embed_file_path.csv.
        data_root: Override; default is get_data_root().

    Returns:
        Absolute path to the file.
    """
    p = path.strip()
    if os.path.isabs(p):
        return p
    root = data_root if data_root is not None else get_data_root()
    return os.path.join(root, p)


def resolve_scdata_paths_df(df):
    """
    Resolve the 'file_path' column of a scdata or perturb_embed DataFrame.
    Returns a copy of the DataFrame with absolute paths.
    """
    if df is None or "file_path" not in df.columns:
        return df
    root = get_data_root()
    out = df.copy()
    out["file_path"] = out["file_path"].apply(lambda p: resolve_data_path(str(p).strip(), root))
    return out
