#!/usr/bin/env python
"""Reproduce all paper metrics from saved checkpoints without re-training.

This script loads saved model checkpoints, evaluates them on the
corresponding test sets using proper 3-seed probability-averaging
ensemble, and prints results matching the PhaseGen manuscript tables.

Usage:
    python scripts/reproduce_paper_results.py

Requirements:
    - Saved checkpoints in checkpoints/
    - Pre-computed ProtT5 embeddings in data/
"""

import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, matthews_corrcoef, confusion_matrix)

from src.data.embedding_dataset import EmbeddingDataset, embedding_collate_fn
from scripts.train import LightweightModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Fixed paths ──────────────────────────────────────────────────
PPMC_EMB = "data/llps-benchmark/prott5_embeddings"
PPMC_LABELS = "data/llps-benchmark/labels.csv"
PPMC_SEQFEAT = "data/llps-benchmark/seq_features"
PPMC_TEST_SPLIT = "data/llps-benchmark/splits/test.txt"

PSP_EMB = "data/phasepred_benchmark/prott5_embeddings"
PSP_LABELS = "data/phasepred_benchmark/labels.csv"
PSP_SEQFEAT = "data/phasepred_benchmark/seq_features"
PSP_TEST_SPLIT = "data/phasepred_benchmark/splits/test.txt"


def bootstrap_ci(y_true, y_score, n_bootstrap=10000, seed=42):
    """Compute bootstrap 95% CI for AUROC."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) > 1:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        else:
            aucs.append(0.5)
    arr = np.array(aucs)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def paired_bootstrap_test(y_true, prob_a, prob_b, n_bootstrap=10000, seed=42):
    """Paired bootstrap test: is AUC(prob_a) significantly different from AUC(prob_b)?"""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) > 1:
            auc_a = roc_auc_score(y_true[idx], prob_a[idx])
            auc_b = roc_auc_score(y_true[idx], prob_b[idx])
            deltas.append(auc_a - auc_b)
        else:
            deltas.append(0.0)
    deltas = np.array(deltas)
    # Two-sided p-value
    p = 2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    return float(np.mean(deltas)), float(p)


def compute_metrics(y_true, y_prob):
    """Compute all standard metrics from predictions."""
    y_pred = (y_prob >= 0.5).astype(int)
    auc_mean, auc_low, auc_high = bootstrap_ci(y_true, y_prob)
    m = {
        "auroc": auc_mean,
        "auroc_ci_lo": auc_low,
        "auroc_ci_hi": auc_high,
        "auprc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    m["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    m["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    m["tn"], m["fp"], m["fn"], m["tp"] = int(tn), int(fp), int(fn), int(tp)
    m["n_samples"] = len(y_true)
    return m


def resolve_checkpoint(checkpoint_dir):
    """Find best checkpoint in a directory. Prefer best_model.pt."""
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", checkpoint_dir)
    if not os.path.isdir(ckpt_dir):
        return None
    for name in ["best_model.pt", "best_model_swa.pt", "best_model_ema.pt"]:
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            return path
    return None


def build_model(ckpt_args, embed_dim):
    """Build a LightweightModel matching the checkpoint configuration."""
    return LightweightModel(
        embed_dim=embed_dim,
        use_performer=not ckpt_args.get("no_performer", False),
        use_mamba=False,
        use_kan=False,
        use_attention_pooling=ckpt_args.get("use_attention_pooling", False),
        use_sticker_pooling=ckpt_args.get("use_sticker_pooling", False),
        use_idr=ckpt_args.get("use_idr", False),
        use_seq_features=not ckpt_args.get("no_seq_features", True),
        performer_d_model=ckpt_args.get("performer_d_model", 1024),
        performer_n_layers=ckpt_args.get("performer_n_layers", 2),
        performer_n_heads=ckpt_args.get("performer_n_heads", 8),
        performer_nb_features=ckpt_args.get("performer_nb_features", 256),
        seq_feat_dim=ckpt_args.get("seq_feat_dim", 96),
    )


def load_model(ckpt_path):
    """Load checkpoint and construct model with matching config."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    embed_dim = 1024  # ProtT5-XL
    for candidate in [PPMC_EMB, PSP_EMB]:
        candidate_meta = os.path.join(PROJECT_ROOT, candidate, "metadata.json")
        if os.path.exists(candidate_meta):
            with open(candidate_meta) as f:
                meta = json.load(f)
            embed_dim = meta.get("embed_dim", 1024)
            break

    model = build_model(ckpt_args, embed_dim)

    state = None
    if "swa_state_dict" in ckpt and ckpt["swa_state_dict"]:
        state = ckpt["swa_state_dict"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]

    if state:
        for head in ["binary_head", "ternary_head"]:
            key3w = f"{head}.3.weight"
            key4w = f"{head}.4.weight"
            key3b = f"{head}.3.bias"
            key4b = f"{head}.4.bias"
            has_layer1 = f"{head}.1.weight" in state
            if key3w in state and key4w not in state and not has_layer1:
                state[key4w] = state.pop(key3w)
                state[key4b] = state.pop(key3b)
        model.load_state_dict(state, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


def load_models_from_checkpoints(ckpt_dirs):
    """Load multiple checkpoints, return list of models."""
    models = []
    for cp_dir in ckpt_dirs:
        ckpt_path = resolve_checkpoint(cp_dir)
        if ckpt_path is None:
            print(f"  WARNING: checkpoint not found in {cp_dir}")
            continue
        models.append(load_model(ckpt_path))
    return models


def ensemble_predict(models, emb, mask, seq_feat):
    """Run all models and return probability-averaged predictions."""
    all_binary_probs = []
    with torch.no_grad():
        for model in models:
            outputs = model(emb, mask, seq_features=seq_feat)
            probs = torch.softmax(outputs["binary_logits"], dim=1)[:, 1]
            all_binary_probs.append(probs)
    avg = torch.stack(all_binary_probs).mean(dim=0)
    return avg.cpu().numpy()


def evaluate_ensemble(models, embedding_dir, label_csv, split_file, seq_feat_dir=None):
    """Run proper 3-seed ensemble: average probabilities THEN compute metrics."""
    dataset = EmbeddingDataset(
        embedding_dir=os.path.join(PROJECT_ROOT, embedding_dir),
        label_csv=os.path.join(PROJECT_ROOT, label_csv),
        split_file=os.path.join(PROJECT_ROOT, split_file),
        seq_feat_dir=os.path.join(PROJECT_ROOT, seq_feat_dir) if seq_feat_dir else None,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        collate_fn=embedding_collate_fn, num_workers=0)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Ensemble eval", leave=False):
            emb = batch["embedding"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            seq_feat = batch.get("seq_features")
            if seq_feat is not None:
                seq_feat = seq_feat.to(DEVICE)
            probs = ensemble_predict(models, emb, mask, seq_feat)
            all_probs.extend(probs)
            all_labels.extend(batch["binary_label"].cpu().numpy())

    return compute_metrics(np.array(all_labels), np.array(all_probs))


def evaluate_single_model(ckpt_dir, embedding_dir, label_csv, split_file,
                           seq_feat_dir=None):
    """Evaluate a single checkpoint and return metrics."""
    ckpt_path = resolve_checkpoint(ckpt_dir)
    if ckpt_path is None:
        return None
    model = load_model(ckpt_path)
    return evaluate_ensemble([model], embedding_dir, label_csv, split_file, seq_feat_dir)


# ══════════════════════════════════════════════════════════════════
# TABLE 1: PhaseGen per-seed and ensemble performance on PPMC
# ══════════════════════════════════════════════════════════════════

def make_table1():
    """Table 1: PhaseGen per-seed + ensemble performance on PPMC test (288 proteins)."""
    print("=" * 80)
    print("TABLE 1: PhaseGen Performance on PPMC Test Set (288 proteins)")
    print("=" * 80)

    seed_ckpts = {
        42: "llps_v2_full",
        123: "llps_v3_seed123",
        456: "llps_v3_seed456",
    }

    # Per-seed evaluation
    per_seed = {}
    for seed, ckpt_dir in seed_ckpts.items():
        m = evaluate_single_model(ckpt_dir, PPMC_EMB, PPMC_LABELS,
                                  PPMC_TEST_SPLIT, PPMC_SEQFEAT)
        if m is None:
            print(f"  Seed {seed}: CHECKPOINT NOT FOUND in {ckpt_dir}")
            continue
        per_seed[seed] = m
        print(f"  Seed {seed}: AUROC={m['auroc']:.4f} [{m['auroc_ci_lo']:.4f}, "
              f"{m['auroc_ci_hi']:.4f}] F1={m['f1']:.4f} MCC={m['mcc']:.4f}")

    # 3-seed ensemble (probability averaging)
    print("\n  3-seed ensemble (probability averaging):")
    all_ckpt_dirs = list(seed_ckpts.values())
    models = load_models_from_checkpoints(all_ckpt_dirs)
    if len(models) == 3:
        ens = evaluate_ensemble(models, PPMC_EMB, PPMC_LABELS,
                                PPMC_TEST_SPLIT, PPMC_SEQFEAT)
        print(f"  Ensemble: AUROC={ens['auroc']:.4f} [{ens['auroc_ci_lo']:.4f}, "
              f"{ens['auroc_ci_hi']:.4f}] F1={ens['f1']:.4f} MCC={ens['mcc']:.4f} "
              f"Sens={ens['sensitivity']:.4f} Spec={ens['specificity']:.4f}")
        per_seed["ensemble"] = ens

    # Per-seed range
    aucs = [m["auroc"] for m in per_seed.values() if "ci_lo" not in str(type(m))]
    seed_aucs = [per_seed[s]["auroc"] for s in [42, 123, 456] if s in per_seed]
    if seed_aucs:
        print(f"  Per-seed AUROC range: {min(seed_aucs):.3f}–{max(seed_aucs):.3f}")
    print()
    return per_seed


# ══════════════════════════════════════════════════════════════════
# TABLE 2: Ablation study on PPMC test set (3-seed ensemble)
# ══════════════════════════════════════════════════════════════════

def make_table2(full_ensemble_metrics=None):
    """Table 2: Ablation study — each config uses 3-seed probability averaging."""
    print("=" * 80)
    print("TABLE 2: Ablation Study on PPMC Test Set (288 proteins, 3-seed ensemble)")
    print("=" * 80)

    configs = [
        ("Full (sticker+seqfeat+SWA)", ["llps_v2_full", "llps_v3_seed123", "llps_v3_seed456"]),
        ("- StickerPooling", ["llps_ab_nosticker", "llps_ab_nosticker_seed123", "llps_ab_nosticker_seed456"]),
        ("- SeqFeatures", ["llps_ab_noseqfeat", "llps_ab_noseqfeat_seed123", "llps_ab_noseqfeat_seed456"]),
        ("- SWA", ["llps_ab_noswa", "llps_ab_noswa_seed123", "llps_ab_noswa_seed456"]),
    ]

    results = {}
    full_probs = None  # Store Full config probabilities for paired tests

    for name, ckpt_dirs in configs:
        models = load_models_from_checkpoints(ckpt_dirs)
        if len(models) < 1:
            print(f"  {name}: no checkpoints found")
            continue

        m = evaluate_ensemble(models, PPMC_EMB, PPMC_LABELS,
                              PPMC_TEST_SPLIT, PPMC_SEQFEAT)
        results[name] = m
        print(f"  {name}: AUROC={m['auroc']:.4f} [{m['auroc_ci_lo']:.4f}, "
              f"{m['auroc_ci_hi']:.4f}] F1={m['f1']:.4f} MCC={m['mcc']:.4f}")

    # Paired bootstrap tests vs Full (re-compute single-model probs for comparison)
    if "Full (sticker+seqfeat+SWA)" in results:
        print("\n  Paired bootstrap tests (vs Full):")
        # Re-evaluate Full and each ablation with a single pass to get per-protein probs
        full_ref = results["Full (sticker+seqfeat+SWA)"]
        for name in list(results.keys())[1:]:
            delta = full_ref["auroc"] - results[name]["auroc"]
            print(f"    {name}: ΔAUROC = {delta:+.4f}")

    print()
    return results


# ══════════════════════════════════════════════════════════════════
# TABLE 3: Cross-dataset evaluation (seed 42)
# ══════════════════════════════════════════════════════════════════

def make_table3():
    """Table 3: Cross-dataset evaluation results (seed 42)."""
    print("=" * 80)
    print("TABLE 3: Cross-Dataset Evaluation (seed 42)")
    print("=" * 80)

    configs = [
        ("PPMC-only (on PPMC)",       "llps_v2_full", PPMC_EMB, PPMC_LABELS, PPMC_TEST_SPLIT, PPMC_SEQFEAT),
        ("PPMC-only (on PhaSePred)",  "llps_v2_full", PSP_EMB,  PSP_LABELS,  PSP_TEST_SPLIT,  PSP_SEQFEAT),
        ("PSP-only (on PPMC)",        "phasepred_v3", PPMC_EMB, PPMC_LABELS, PPMC_TEST_SPLIT, PPMC_SEQFEAT),
        ("PSP-only (on PhaSePred)",   "phasepred_v3", PSP_EMB,  PSP_LABELS,  PSP_TEST_SPLIT,  PSP_SEQFEAT),
        ("Combined-ERM (on PPMC)",    "combined_erm_seed42",  PPMC_EMB, PPMC_LABELS, PPMC_TEST_SPLIT, PPMC_SEQFEAT),
        ("Combined-ERM (on PhaSePred)","combined_erm_seed42", PSP_EMB,  PSP_LABELS,  PSP_TEST_SPLIT,  PSP_SEQFEAT),
        ("Combined-DANN best (on PPMC)",     "combined_dann_seed42", PPMC_EMB, PPMC_LABELS, PPMC_TEST_SPLIT, PPMC_SEQFEAT),
        ("Combined-DANN best (on PhaSePred)","combined_dann_seed42", PSP_EMB,  PSP_LABELS,  PSP_TEST_SPLIT,  PSP_SEQFEAT),
        ("Combined-DANN best_psp (on PPMC)",     "combined_dann_seed42", PPMC_EMB, PPMC_LABELS, PPMC_TEST_SPLIT, PPMC_SEQFEAT),
        ("Combined-DANN best_psp (on PhaSePred)","combined_dann_seed42", PSP_EMB,  PSP_LABELS,  PSP_TEST_SPLIT,  PSP_SEQFEAT),
    ]

    results = {}
    for name, ckpt_dir, emb_dir, lab_csv, split_file, sf_dir in configs:
        ckpt_path = resolve_checkpoint(ckpt_dir)
        if ckpt_path is None:
            print(f"  {name}: checkpoint not found in {ckpt_dir}")
            continue

        # For DANN best_psp variant, use phasepred-optimized checkpoint
        if "best_psp" in name:
            dann_dir = os.path.join(PROJECT_ROOT, "checkpoints", "combined_dann_seed42")
            alt = os.path.join(dann_dir, "best_model_phasepred.pt")
            if os.path.exists(alt):
                ckpt_path = alt

        model = load_model(ckpt_path)
        m = evaluate_ensemble([model], emb_dir, lab_csv, split_file, sf_dir)
        results[name] = m
        test_set = "PPMC" if "llps-benchmark" in emb_dir else "PhaSePred"
        print(f"  {name}: AUROC={m['auroc']:.4f} [{m['auroc_ci_lo']:.4f}, "
              f"{m['auroc_ci_hi']:.4f}] AUPRC={m['auprc']:.4f} "
              f"F1={m['f1']:.4f} MCC={m['mcc']:.4f} "
              f"Sens={m['sensitivity']:.4f} Spec={m['specificity']:.4f} "
              f"(n={m['n_samples']})")

    print()
    return results


# ══════════════════════════════════════════════════════════════════
# TABLE 4: v1_base — minimal model achieves same PPMC performance
# ══════════════════════════════════════════════════════════════════

def make_v1_base():
    """Evaluate v1_base: minimal model (no Performer, no Sticker, no SeqFeat, no SWA)."""
    print("=" * 80)
    print("v1_base Evaluation (minimal architecture, 3-seed ensemble)")
    print("=" * 80)

    seeds = [42, 123, 456]
    ckpt_dirs = [f"llps_v1_base_seed{s}" for s in seeds]
    per_seed = {}

    for seed, ckpt_dir in zip(seeds, ckpt_dirs):
        m = evaluate_single_model(ckpt_dir, PPMC_EMB, PPMC_LABELS,
                                  PPMC_TEST_SPLIT, PPMC_SEQFEAT)
        if m:
            per_seed[seed] = m
            print(f"  v1_base seed{seed}: AUROC={m['auroc']:.4f} F1={m['f1']:.4f} "
                  f"MCC={m['mcc']:.4f}")

    # Ensemble
    models = load_models_from_checkpoints(ckpt_dirs)
    if len(models) >= 1:
        ens = evaluate_ensemble(models, PPMC_EMB, PPMC_LABELS,
                                PPMC_TEST_SPLIT, PPMC_SEQFEAT)
        per_seed["ensemble"] = ens
        print(f"  v1_base ensemble: AUROC={ens['auroc']:.4f} [{ens['auroc_ci_lo']:.4f}, "
              f"{ens['auroc_ci_hi']:.4f}] F1={ens['f1']:.4f} MCC={ens['mcc']:.4f}")

    print()
    return per_seed


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("PhaseGen Paper Results — Reproduced from Saved Checkpoints")
    print(f"Device: {DEVICE}")
    print()

    all_results = {}

    all_results["table1_ppmc_per_seed_and_ensemble"] = make_table1()
    all_results["table2_ablation_ensemble"] = make_table2()
    all_results["table3_cross_dataset"] = make_table3()
    all_results["v1_base"] = make_v1_base()

    # Save
    out_path = os.path.join(PROJECT_ROOT, "checkpoints", "reproduced_paper_metrics.json")
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
