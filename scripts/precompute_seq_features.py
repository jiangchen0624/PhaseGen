#!/usr/bin/env python
"""Pre-compute biophysical sequence features for all proteins in a FASTA file.

These 6 scalar features are computed offline and cached alongside embeddings.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tqdm import tqdm

from src.utils.sequence_features import compute_sequence_features, features_to_tensor


def parse_fasta(path: str) -> dict:
    """Parse a FASTA file, returning {uniprot_id: sequence}."""
    seqs = {}
    current_id = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]  # Take first token as ID
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            seqs[current_id] = "".join(current_seq)
    return seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str,
                        default="D:/Users/朱江龙/phasegen/data/llps-benchmark/sequences.fasta")
    parser.add_argument("--out-dir", type=str,
                        default="D:/Users/朱江龙/phasegen/data/llps-benchmark/seq_features")
    parser.add_argument("--split-dir", type=str,
                        default="D:/Users/朱江龙/phasegen/data/llps-benchmark/splits")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip Z-score normalization")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seqs = parse_fasta(args.fasta)
    print(f"Loaded {len(seqs)} sequences from {args.fasta}")

    # Compute raw features
    all_features = {}
    for uid, seq in tqdm(seqs.items(), desc="Computing features"):
        feats = compute_sequence_features(seq)
        all_features[uid] = feats

    feature_names = ["aromatic_fraction", "net_charge", "scd", "kappa", "fcr", "log_length"]

    # Compute normalization stats from training set only
    if not args.no_normalize:
        import numpy as np
        train_ids = set()
        train_file = os.path.join(args.split_dir, "train.txt")
        if os.path.exists(train_file):
            with open(train_file) as f:
                train_ids = set(l.strip() for l in f if l.strip())
            print(f"Loaded {len(train_ids)} training IDs for normalization")

        if train_ids:
            train_feats = {uid: all_features[uid] for uid in train_ids if uid in all_features}
            print(f"Computing normalization stats from {len(train_feats)} training proteins")
            norm_stats = {}
            for name in feature_names:
                vals = np.array([train_feats[uid][name] for uid in train_feats])
                norm_stats[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        else:
            # Fallback: use all proteins
            print("WARNING: No train split found, using all proteins for normalization")
            norm_stats = {}
            for name in feature_names:
                vals = np.array([all_features[uid][name] for uid in all_features])
                norm_stats[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        # Save normalization stats
        with open(os.path.join(args.out_dir, "norm_stats.json"), "w") as f:
            json.dump(norm_stats, f, indent=2)
        print("\nNormalization stats (from training set):")
        for name in feature_names:
            print(f"  {name:20s}: mean={norm_stats[name]['mean']:+.4f}  std={norm_stats[name]['std']:.4f}")

    # Save normalized features
    import numpy as np
    for uid in tqdm(all_features, desc="Saving features"):
        feats = all_features[uid]
        if not args.no_normalize:
            normed = {}
            for name in feature_names:
                m = norm_stats[name]["mean"]
                s = norm_stats[name]["std"] + 1e-8
                normed[name] = np.clip((feats[name] - m) / s, -5.0, 5.0)
            tensor = features_to_tensor(normed)
        else:
            tensor = features_to_tensor(feats)
        torch.save(tensor, os.path.join(args.out_dir, f"{uid}.pt"))

    # Print raw statistics
    print("\nRaw feature statistics across all proteins:")
    for name in feature_names:
        vals = [all_features[uid][name] for uid in all_features]
        print(f"  {name:20s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    print(f"\nSaved {len(all_features)} normalized feature files to {args.out_dir}")


if __name__ == "__main__":
    main()
