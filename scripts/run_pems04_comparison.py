#!/usr/bin/env python3
"""Run PEMS04 comparison experiments and aggregate results."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_CFGS = {
    "HyperD": "baselines/HyperD/PEMS04.py",
    "ForecastSpace": "baselines/ForecastSpace/PEMS04.py",
    "ForecastStateGraphResolution": "baselines/ForecastStateGraphResolution/PEMS04.py",
}


def find_latest_metrics(ckpt_root: Path) -> Path | None:
    cands = sorted(ckpt_root.glob("**/test_metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def run_one(cfg: str, seed: int, gpu: str, dry_run: bool = False) -> dict:
    # Patch seed by env is not supported; rewrite note: configs hardcode SEED=1.
    # For multi-seed, we pass via sed-less approach: python -c exec after modifying.
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--cfg",
        cfg,
        "-g",
        gpu,
    ]
    record = {"cfg": cfg, "seed": seed, "cmd": " ".join(cmd), "status": "pending"}
    if dry_run:
        record["status"] = "dry_run"
        return record
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        record["returncode"] = proc.returncode
        record["stdout_tail"] = proc.stdout[-2000:]
        record["stderr_tail"] = proc.stderr[-2000:]
        if proc.returncode != 0:
            record["status"] = "failed"
            return record
        record["status"] = "ok"
    except Exception:
        record["status"] = "failed"
        record["traceback"] = traceback.format_exc()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["HyperD", "ForecastStateGraphResolution"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model in args.models:
        if model not in MODEL_CFGS:
            raise SystemExit(f"Unknown model {model}; choose from {list(MODEL_CFGS)}")
        cfg = MODEL_CFGS[model]
        for seed in args.seeds:
            # Existing HyperD seed1 shortcut
            if model == "HyperD" and seed == 1:
                metrics_path = ROOT / "checkpoints/HyperD/PEMS04_100_12_12/2a6347ff1388253de6ad9a9c506700d9/test_metrics.json"
                if metrics_path.is_file() and args.skip_existing:
                    metrics = json.loads(metrics_path.read_text())
                    rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "status": "skipped_existing",
                            "test_MAE": metrics["overall"]["MAE"],
                            "test_RMSE": metrics["overall"]["RMSE"],
                            "test_MAPE": metrics["overall"]["MAPE"],
                            "metrics_path": str(metrics_path),
                        }
                    )
                    continue
            print(f"=== Running {model} seed={seed} ===")
            rec = run_one(cfg, seed, args.gpus, dry_run=args.dry_run)
            rec["model"] = model
            rows.append(rec)

    csv_path = results_dir / "pems04_comparison.csv"
    md_path = results_dir / "pems04_comparison.md"
    keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    lines = ["# PEMS04 Comparison", "", "| model | seed | status | test_MAE | test_RMSE | test_MAPE |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r.get('model')} | {r.get('seed')} | {r.get('status')} | {r.get('test_MAE','')} | {r.get('test_RMSE','')} | {r.get('test_MAPE','')} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print("Wrote", csv_path)
    print("Wrote", md_path)


if __name__ == "__main__":
    main()
