"""Unified CLI entry-point for hippewav2vec2 pipeline stages.

Usage:
    python -m homewav2vec2.cli ingest    --config src/homewav2vec2/config/data.yaml
    python -m homewav2vec2.cli validate  --config src/homewav2vec2/config/data.yaml
    python -m homewav2vec2.cli transform --config src/homewav2vec2/config/data.yaml
    python -m homewav2vec2.cli push_hf   --config src/homewav2vec2/config/training.yaml
"""

from __future__ import annotations

import argparse
import sys

import yaml

from homewav2vec2.utils.logging import setup_logging


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def cmd_ingest(args: argparse.Namespace) -> None:
    from homewav2vec2.data.ingest import run_ingestion

    cfg = _load_yaml(args.config)
    run_ingestion(cfg)


def cmd_validate(args: argparse.Namespace) -> None:
    from homewav2vec2.data.validate import run_validation

    cfg = _load_yaml(args.config)
    run_validation(cfg)


def cmd_transform(args: argparse.Namespace) -> None:
    from homewav2vec2.data.transform import run_transformation

    cfg = _load_yaml(args.config)
    run_transformation(cfg)


def cmd_push_hf(args: argparse.Namespace) -> None:
    from homewav2vec2.utils.env import load_env, get_hf_token, get_hf_repo_id, get_hf_private

    load_env()
    token = get_hf_token()
    repo_id = get_hf_repo_id()
    private = get_hf_private()

    cfg = _load_yaml(args.config)
    model_dir = cfg.get("output_dir", "artifacts/training") + "/final_model"

    from pathlib import Path

    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"ERROR: model directory not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        commit_message="Upload wav2vec2 home-domain SSL model",
    )
    print(f"Model pushed to https://huggingface.co/{repo_id}")


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(prog="homewav2vec2", description="HindiBabyNet Wav2Vec2 SSL pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Phase 1: build raw audio manifest")
    p_ingest.add_argument("--config", required=True, help="Path to data.yaml")

    # validate
    p_val = sub.add_parser("validate", help="Phase 2: validate audio files")
    p_val.add_argument("--config", required=True, help="Path to data.yaml")

    # transform
    p_trans = sub.add_parser("transform", help="Phase 3: create train/dev splits")
    p_trans.add_argument("--config", required=True, help="Path to data.yaml")

    # push_hf
    p_push = sub.add_parser("push_hf", help="Phase 5: push model to HF Hub")
    p_push.add_argument("--config", required=True, help="Path to training.yaml")

    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "validate": cmd_validate,
        "transform": cmd_transform,
        "push_hf": cmd_push_hf,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
