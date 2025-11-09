#!/usr/bin/env python3
"""
Automated experiment runner for Contrast2Contrast ablation studies.

Usage:
    # Run all experiments
    python run_experiments.py --all

    # Run specific experiment groups
    python run_experiments.py --cross-domain
    python run_experiments.py --loss-ablation

    # Run specific configs
    python run_experiments.py --config configs/cross_no_skip.json

    # Dry run (show what would be run)
    python run_experiments.py --all --dry-run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict

# Define experiment groups
CROSS_DOMAIN_EXPERIMENTS = [
    "configs/cross_baseline.json",
    "configs/cross_no_skip.json",
    "configs/cross_target_skip.json",
    "configs/cross_zero_skip.json",
    "configs/cross_mixed_skip.json",
]

LOSS_ABLATION_EXPERIMENTS = [
    "configs/ablation_all_losses.json",
    "configs/ablation_no_content.json",
    "configs/ablation_no_recon.json",
    "configs/ablation_no_cross.json",
    "configs/ablation_only_cross.json",
    "configs/ablation_content_recon.json",
    "configs/ablation_strong_edge.json",
]

ALL_EXPERIMENTS = CROSS_DOMAIN_EXPERIMENTS + LOSS_ABLATION_EXPERIMENTS


def run_experiment(config_path: str, epochs: int = 10, dry_run: bool = False) -> bool:
    """Run a single experiment with the given config."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"❌ Config not found: {config_path}")
        return False

    cmd = ["python", "train.py", "--config", str(config_file), "--epochs", str(epochs)]

    print(f"\n{'='*80}")
    print(f"🚀 Running experiment: {config_file.stem}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("   [DRY RUN - not actually executing]")
        return True

    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✅ Completed {config_file.stem} in {elapsed:.1f}s")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed {config_file.stem}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted {config_file.stem}")
        raise


def print_experiment_summary():
    """Print summary of all available experiments."""
    print("\n" + "="*80)
    print("📋 EXPERIMENT SUMMARY")
    print("="*80)

    print(f"\n🔬 Cross-Domain Strategy Experiments ({len(CROSS_DOMAIN_EXPERIMENTS)} configs):")
    for i, config in enumerate(CROSS_DOMAIN_EXPERIMENTS, 1):
        name = Path(config).stem.replace("cross_", "")
        print(f"   {i}. {name:20s} - {config}")

    print(f"\n🧪 Loss Ablation Experiments ({len(LOSS_ABLATION_EXPERIMENTS)} configs):")
    for i, config in enumerate(LOSS_ABLATION_EXPERIMENTS, 1):
        name = Path(config).stem.replace("ablation_", "")
        print(f"   {i}. {name:20s} - {config}")

    print(f"\n📊 Total: {len(ALL_EXPERIMENTS)} experiments\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Contrast2Contrast ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--all", action="store_true",
                        help="Run all experiments")
    parser.add_argument("--cross-domain", action="store_true",
                        help="Run cross-domain strategy experiments")
    parser.add_argument("--loss-ablation", action="store_true",
                        help="Run loss ablation experiments")
    parser.add_argument("--config", type=str,
                        help="Run specific config file")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--summary", action="store_true",
                        help="Print experiment summary and exit")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue running even if an experiment fails")

    args = parser.parse_args()

    if args.summary:
        print_experiment_summary()
        return 0

    # Determine which experiments to run
    experiments: List[str] = []

    if args.config:
        experiments = [args.config]
    elif args.all:
        experiments = ALL_EXPERIMENTS
    elif args.cross_domain:
        experiments = CROSS_DOMAIN_EXPERIMENTS
    elif args.loss_ablation:
        experiments = LOSS_ABLATION_EXPERIMENTS
    else:
        parser.print_help()
        print("\n⚠️  No experiments selected. Use --all, --cross-domain, --loss-ablation, or --config")
        return 1

    # Print summary
    print_experiment_summary()
    print(f"{'='*80}")
    print(f"🎯 Selected {len(experiments)} experiment(s) to run")
    print(f"⏱️  Epochs per experiment: {args.epochs}")
    if args.dry_run:
        print("🏃 Mode: DRY RUN (no actual execution)")
    print(f"{'='*80}\n")

    if not args.dry_run:
        response = input("Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            return 0

    # Run experiments
    results: Dict[str, bool] = {}
    failed_experiments: List[str] = []

    try:
        for i, config in enumerate(experiments, 1):
            print(f"\n{'#'*80}")
            print(f"# Experiment {i}/{len(experiments)}")
            print(f"{'#'*80}")

            success = run_experiment(config, epochs=args.epochs, dry_run=args.dry_run)
            results[config] = success

            if not success:
                failed_experiments.append(config)
                if not args.continue_on_error and not args.dry_run:
                    print(f"\n⚠️  Stopping due to failure. Use --continue-on-error to continue.")
                    break

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment suite interrupted by user")

    # Print final summary
    print(f"\n\n{'='*80}")
    print("📊 FINAL RESULTS")
    print(f"{'='*80}")

    successful = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\n✅ Successful: {successful}/{total}")
    if failed_experiments:
        print(f"❌ Failed: {len(failed_experiments)}/{total}")
        for exp in failed_experiments:
            print(f"   - {exp}")

    print(f"\n{'='*80}\n")

    return 0 if not failed_experiments else 1


if __name__ == "__main__":
    sys.exit(main())
