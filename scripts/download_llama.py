"""Download the configured LLaMA-family model into the local CONVERA project."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=os.getenv("CONVERA_MODEL_REPO", "meta-llama/Meta-Llama-3-8B"))
    parser.add_argument("--local-dir", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    target = Path(args.local_dir)
    target.mkdir(parents=True, exist_ok=True)
    print(f"[CONVERA] downloading {args.repo_id} -> {target}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=target,
        token=os.getenv("HF_TOKEN") or None,
    )
    print("[CONVERA] model download complete")


if __name__ == "__main__":
    main()
