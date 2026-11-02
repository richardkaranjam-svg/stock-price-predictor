"""
batch_processor.py — One-Click Automated Strategy Ingestion & Packaging Pipeline.

Processes any new strategy CSV dropped into `inbox/`:
1. Standardizes columns and runs baseline audit.
2. Runs ML Meta-Labeling optimization.
3. Regenerates the Master Strategy Leaderboard.
4. Packages a zero-friction client deliverable (.zip) in `outbox/`.

Usage:
    .venv\\Scripts\\python.exe batch_processor.py
    .venv\\Scripts\\python.exe batch_processor.py --inbox custom_inbox --outbox custom_outbox
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

from audit_strategy import run_strategy_backtest
from meta_label_optimizer import run_meta_labeling_optimization
from generate_league_table import generate_master_league_table


def process_inbox(inbox_dir: str = "inbox", outbox_dir: str = "outbox") -> list[dict]:
    in_path = Path(inbox_dir)
    out_path = Path(outbox_dir)
    in_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(in_path.glob("*.csv"))
    if not csv_files:
        print(f"\nNo CSV files found in '{in_path}/'. Drop some trade logs in there to process!")
        return []

    print("\n" + "=" * 70)
    print(f"       PROCESSING {len(csv_files)} STRATEGY FILE(S) FROM INBOX        ")
    print("=" * 70)

    results = []

    for csv_file in csv_files:
        strat_name = csv_file.stem.strip()
        print(f"\n>>> Processing Strategy: [{strat_name}] from {csv_file.name}")

        strat_report_dir = Path("reports") / "inbox_audits" / strat_name
        meta_report_dir = Path("reports") / "inbox_audits" / f"{strat_name}_meta_optimized"

        # 1. Run Baseline Strategy Audit
        print("  [1/4] Running baseline strategy audit...")
        base_res = run_strategy_backtest(str(csv_file), output_dir=str(strat_report_dir))
        print(f"        Verdict: {base_res['verdict']} | PF: {base_res['metrics']['profit_factor']:.2f} | Win Rate: {base_res['metrics']['win_rate']*100:.1f}%")

        # 2. Run ML Meta-Labeling Optimizer if confidence exists
        has_meta = False
        try:
            print("  [2/4] Testing ML Meta-Labeling optimization...")
            meta_res = run_meta_labeling_optimization(str(csv_file), output_dir=str(meta_report_dir))
            print(f"        Optimized: Conf >= {meta_res['optimal_threshold']:.2f} | PF: {meta_res['optimal_metrics']['profit_factor']:.2f} | Win Rate: {meta_res['optimal_metrics']['win_rate']*100:.1f}%")
            has_meta = True
        except Exception as e:
            print(f"        Notice: Meta-labeling skipped ({e})")

        # 3. Create Client Deliverable ZIP Package in outbox/
        print("  [3/4] Packaging client deliverables into ZIP...")
        zip_filename = out_path / f"{strat_name}_Audit_Package.zip"
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add base audit files
            for f in strat_report_dir.glob("*.*"):
                zf.write(f, arcname=f"baseline_audit/{f.name}")
            # Add meta-labeled files if available
            if has_meta and meta_report_dir.is_dir():
                for f in meta_report_dir.glob("*.*"):
                    zf.write(f, arcname=f"ml_meta_optimized/{f.name}")

        print(f"        Package generated: {zip_filename} ({zip_filename.stat().st_size / 1024:.1f} KB)")
        results.append({
            "strategy": strat_name,
            "verdict": base_res["verdict"],
            "zip_package": str(zip_filename),
        })

    # 4. Regenerate Master League Table to include new audits
    print("\n  [4/4] Updating Master Strategy Leaderboard...")
    league_file = generate_master_league_table("reports", "reports/league_table.html")
    print(f"        Leaderboard updated: {league_file}")

    print("\n" + "=" * 70)
    print("                      ALL BATCH AUDITS COMPLETE                       ")
    print("=" * 70)
    for r in results:
        print(f"  * {r['strategy']:30s} -> [{r['verdict']}] -> {r['zip_package']}")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process strategy CSVs from inbox into outbox")
    parser.add_argument("--inbox", type=str, default="inbox", help="Input directory containing CSV trade logs")
    parser.add_argument("--outbox", type=str, default="outbox", help="Output directory for zipped deliverable packages")
    args = parser.parse_args()

    process_inbox(args.inbox, args.outbox)
