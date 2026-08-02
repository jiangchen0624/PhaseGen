import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, confusion_matrix, accuracy_score
)
from typing import Dict


def compute_metrics(binary_logits, binary_labels,
                    ternary_logits = None,
                    ternary_labels = None) -> Dict:
    """Compute comprehensive evaluation metrics.

    Args:
        binary_logits: (N, 2) logits or (N,) probabilities (torch.Tensor or np.ndarray)
        binary_labels: (N,) [0, 1] (torch.Tensor or np.ndarray)
        ternary_logits: optional (N, 3) logits
        ternary_labels: optional (N,) [0, 1, 2]

    Returns dict of metric names to values.
    """
    # Convert to torch tensors if needed
    if isinstance(binary_logits, np.ndarray):
        binary_logits = torch.from_numpy(binary_logits)
    if isinstance(binary_labels, np.ndarray):
        binary_labels = torch.from_numpy(binary_labels)

    binary_probs = torch.softmax(binary_logits.float(), dim=-1).detach().cpu().numpy()
    binary_preds = np.argmax(binary_probs, axis=-1)
    binary_labels_np = binary_labels.detach().cpu().numpy().astype(int)

    metrics = {}

    # Binary metrics
    ps_prob = binary_probs[:, 1]

    # Check for AUROC (needs both classes present)
    if len(np.unique(binary_labels_np)) > 1:
        metrics["binary_auroc"] = float(roc_auc_score(binary_labels_np, ps_prob))
        metrics["binary_auprc"] = float(average_precision_score(binary_labels_np, ps_prob))
    else:
        metrics["binary_auroc"] = 0.5
        metrics["binary_auprc"] = 0.0

    metrics["binary_accuracy"] = float(accuracy_score(binary_labels_np, binary_preds))
    metrics["binary_f1"] = float(f1_score(binary_labels_np, binary_preds, average="binary",
                                           zero_division=0))
    metrics["binary_mcc"] = float(matthews_corrcoef(binary_labels_np, binary_preds))

    # Confusion matrix
    cm = confusion_matrix(binary_labels_np, binary_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["binary_confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        metrics["binary_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["binary_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Ternary metrics (if provided)
    if ternary_logits is not None and ternary_labels is not None:
        ternary_probs = torch.softmax(ternary_logits, dim=-1).detach().cpu().numpy()
        ternary_preds = np.argmax(ternary_probs, axis=-1)
        ternary_labels_np = ternary_labels.detach().cpu().numpy().astype(int)

        metrics["ternary_accuracy"] = float(accuracy_score(ternary_labels_np, ternary_preds))
        metrics["ternary_f1_macro"] = float(f1_score(ternary_labels_np, ternary_preds,
                                                     average="macro", zero_division=0))
        metrics["ternary_mcc"] = float(matthews_corrcoef(ternary_labels_np, ternary_preds))

        # Per-class AUROC (one-vs-rest)
        if len(np.unique(ternary_labels_np)) > 1:
            try:
                metrics["ternary_auroc_ovr"] = float(roc_auc_score(
                    ternary_labels_np, ternary_probs, multi_class="ovr", average="macro"
                ))
            except ValueError:
                metrics["ternary_auroc_ovr"] = 0.5

        # Ternary confusion matrix
        cm3 = confusion_matrix(ternary_labels_np, ternary_preds, labels=[0, 1, 2])
        metrics["ternary_confusion"] = cm3.tolist()

    return metrics


def format_metrics(metrics: Dict) -> str:
    """Format metrics dict as a readable string."""
    lines = []
    for key, value in metrics.items():
        if "confusion" in key:
            continue
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)
