"""One-click pipeline: extract embeddings → compute features → split data → train.

Usage:
    python run_pipeline.py --input data/ppmc/sequences.fasta --labels data/ppmc/labels.csv --name ppmc_exp
"""

import argparse
import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def run_step(description, cmd):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {description}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="PhaseGen one-click pipeline")
    parser.add_argument("--input", required=True, help="Path to input FASTA file (sequences.fasta)")
    parser.add_argument("--labels", required=True, help="Path to labels CSV (id,binary_label,ternary_label)")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--data_dir", default=None, help="Data directory (default: data/<name>)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_extract", action="store_true", help="Skip embedding extraction (if already done)")
    parser.add_argument("--skip_features", action="store_true", help="Skip feature computation")
    parser.add_argument("--skip_splits", action="store_true", help="Skip split creation")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"data/{args.name}"

    os.makedirs(os.path.join(PROJECT_ROOT, args.data_dir, "splits"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)

    if not args.skip_extract:
        run_step("Step 1/4: Extracting ProtT5 embeddings",
                 [PYTHON, "scripts/extract_prott5.py",
                  "--fasta", args.input,
                  "--labels", args.labels,
                  "--output_dir", f"{args.data_dir}/prott5_embeddings"])

    if not args.skip_features:
        run_step("Step 2/4: Computing biophysical features",
                 [PYTHON, "scripts/precompute_seq_features.py",
                  "--fasta", args.input,
                  "--output_dir", f"{args.data_dir}/seq_features"])

    if not args.skip_splits:
        run_step("Step 3/4: Creating data splits",
                 [PYTHON, "scripts/create_splits.py",
                  "--labels", args.labels,
                  "--output_dir", f"{args.data_dir}/splits"])

    run_step("Step 4/4: Training model",
             [PYTHON, "scripts/train.py",
              "--embedding_dir", f"{args.data_dir}/prott5_embeddings",
              "--label_csv", args.labels,
              "--splits_dir", f"{args.data_dir}/splits",
              "--seq_feat_dir", f"{args.data_dir}/seq_features",
              "--output_dir", f"checkpoints/{args.name}",
              "--epochs", str(args.epochs),
              "--batch_size", str(args.batch_size),
              "--device", args.device])

    print(f"\n{'='*60}")
    print(f"  Pipeline done! Checkpoint: checkpoints/{args.name}/best_model.pt")
    print(f"  Evaluate: python scripts/evaluate.py --checkpoint_dir checkpoints/{args.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
