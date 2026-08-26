"""Run a parameter sweep and rank runs by temporal validation PR-AUC.

Example:
python run_experiments.py --train HACKATHON_TRAINING_DATA.CSV \
    --artifacts-root experiments --ae-epochs 40,80 \
    --n-iter 12,25 --latent-dim 16,32 --n-jobs 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    args = parse_args()
    root = Path(args.artifacts_root)
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    combinations = list(itertools.product(args.ae_epochs, args.n_iter, args.latent_dim))
    if args.max_runs:
        combinations = combinations[: args.max_runs]

    for run_number, (ae_epochs, n_iter, latent_dim) in enumerate(combinations, start=1):
        run_dir = root / f"run_{run_number:03d}_ae{ae_epochs}_iter{n_iter}_z{latent_dim}"
        metadata_path = run_dir / "metadata.json"
        command = [
            sys.executable,
            "train_pipeline.py",
            "--train", args.train,
            "--artifacts", str(run_dir),
            "--ae-epochs", str(ae_epochs),
            "--n-iter", str(n_iter),
            "--latent-dim", str(latent_dim),
            "--n-jobs", str(args.n_jobs),
            "--seed", str(args.seed),
        ]
        print(f"\n[{run_number}/{len(combinations)}] {' '.join(command)}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0 or not metadata_path.exists():
            results.append({
                "run": str(run_dir),
                "ae_epochs": ae_epochs,
                "n_iter": n_iter,
                "latent_dim": latent_dim,
                "status": "failed",
                "returncode": completed.returncode,
            })
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        results.append({
            "run": str(run_dir),
            "ae_epochs": ae_epochs,
            "n_iter": n_iter,
            "latent_dim": latent_dim,
            "status": "completed",
            "validation_pr_auc": metadata["validation_pr_auc"],
            "validation_roc_auc": metadata["validation_roc_auc"],
        })

    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    completed_runs = [row for row in results if row["status"] == "completed"]
    if not completed_runs:
        raise SystemExit("No experiment completed successfully.")
    best = max(completed_runs, key=lambda row: row["validation_pr_auc"])
    print("\nBest run by validation PR-AUC:")
    print(json.dumps(best, indent=2))
    print(f"\nAll results saved to: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--artifacts-root", default="experiments")
    parser.add_argument("--ae-epochs", type=int, nargs="+", default=[40, 80])
    parser.add_argument("--n-iter", type=int, nargs="+", default=[12, 25])
    parser.add_argument("--latent-dim", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-runs", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
