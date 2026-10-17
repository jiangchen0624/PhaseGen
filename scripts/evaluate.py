#!/usr/bin/env python
"""Evaluate a trained checkpoint on both PPMC and PhaSePred test sets.

Usage: python scripts/eval_checkpoint_cross.py --checkpoint checkpoints/combined_erm_seed42/best_model.pt
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, matthews_corrcoef)

from src.data.embedding_dataset import EmbeddingDataset, embedding_collate_fn
from scripts.train_fast import LightweightModel

BASE = "D:/Users/朱江龙/phasegen"


def evaluate_checkpoint(checkpoint_path, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Args: {ckpt_args}")

    model = LightweightModel(
        embed_dim=ckpt_args.get("embed_dim", 1024),
        use_performer=not ckpt_args.get("no_performer", False),
        use_mamba=False, use_kan=False,
        use_attention_pooling=False,
        use_idr=False,
        use_sticker_pooling=ckpt_args.get("use_sticker_pooling", True),
        use_seq_features=not ckpt_args.get("no_seq_features", False),
        performer_d_model=ckpt_args.get("performer_d_model", 1024),
        performer_n_layers=ckpt_args.get("performer_n_layers", 2),
        performer_dropout=ckpt_args.get("dropout", 0.1),
        conv_dropout=ckpt_args.get("dropout", 0.1) * 2,
        kan_dropout=ckpt_args.get("dropout", 0.1),
    ).to(device)

    # Load state dict (handle SWA wrapped models)
    state = ckpt.get("swa_model_state", ckpt.get("model_state_dict"))
    if "module." in list(state.keys())[0]:
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    results = {}

    # Evaluate on PPMC test
    for name, emb_dir, label_csv, split_file, seq_feat_dir in [
        ("PPMC_test", f"{BASE}/data/llps-benchmark/prott5_embeddings",
         f"{BASE}/data/llps-benchmark/labels.csv",
         f"{BASE}/data/llps-benchmark/splits/test.txt",
         f"{BASE}/data/llps-benchmark/seq_features"),
        ("PhaSePred_test", f"{BASE}/data/phasepred_benchmark/prott5_embeddings",
         f"{BASE}/data/phasepred_benchmark/test_labels.csv",
         f"{BASE}/data/phasepred_benchmark/test.txt",
         f"{BASE}/data/phasepred_benchmark/seq_features"),
    ]:
        if not os.path.exists(emb_dir) or not os.path.exists(label_csv):
            print(f"  Skipping {name} — data not found")
            continue

        dataset = EmbeddingDataset(
            embedding_dir=emb_dir, label_csv=label_csv,
            split_file=split_file, seq_feat_dir=seq_feat_dir,
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=False,
                          collate_fn=embedding_collate_fn, num_workers=0)
        print(f"\n{name}: {len(dataset)} proteins")

        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=name, leave=False):
                emb = batch["embedding"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["binary_label"]
                seq_feat = batch.get("seq_features")
                if seq_feat is not None:
                    seq_feat = seq_feat.to(device)

                outputs = model(emb, mask, seq_features=seq_feat)
                probs = torch.softmax(outputs["binary_logits"], dim=-1)[:, 1]
                all_probs.append(probs.cpu())
                all_labels.append(labels)

        probs = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy()

        # Youden-optimal threshold from PPMC (or use 0.5)
        if name == "PPMC_test":
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(labels, probs)
            youden_idx = np.argmax(tpr - fpr)
            best_thresh = thresholds[youden_idx]
            print(f"  Youden threshold (PPMC): {best_thresh:.4f}")

        preds = (probs > 0.5).astype(int)
        auc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
        f1 = f1_score(labels, preds, zero_division=0)
        mcc = matthews_corrcoef(labels, preds)

        print(f"  AUROC={auc:.4f}  AUPRC={auprc:.4f}  F1={f1:.4f}  MCC={mcc:.4f}")
        print(f"  Pos={labels.sum()}  Neg={(1-labels).sum()}")

        results[name] = {
            "auroc": float(auc), "auprc": float(auprc),
            "f1": float(f1), "mcc": float(mcc),
            "n_pos": int(labels.sum()), "n_neg": int((1-labels).sum()),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/combined_erm_seed42/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results = evaluate_checkpoint(args.checkpoint, args.device)

    # Save
    out_dir = os.path.dirname(args.checkpoint)
    with open(f"{out_dir}/cross_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON: PPMC-trained vs Combined-ERM")
    print("=" * 60)
    print(f"{'Model':<20} {'PPMC AUROC':>12} {'PhaSePred AUROC':>16}")
    print("-" * 50)
    print(f"{'PPMC-only':<20} {0.966:>12.4f} {0.645:>16.4f}")
    if "PPMC_test" in results and "PhaSePred_test" in results:
        print(f"{'Combined-ERM (new)':<20} {results['PPMC_test']['auroc']:>12.4f} {results['PhaSePred_test']['auroc']:>16.4f}")


if __name__ == "__main__":
    main()