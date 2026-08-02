"""Compute biophysical sequence features known to correlate with phase separation.

Features are based on the sticker-spacer framework (Martin et al. 2020, Science;
Holehouse et al. 2017, BPJ) and related literature:

  - aromatic_fraction: F+Y+W / L  (sticker residues drive PS via pi-pi interactions)
  - net_charge:  (K+R) - (D+E) / L  (net charge modulates PS propensity)
  - scd: Sequence Charge Decoration — how clustered charges are along sequence
  - kappa: (K-R)/(K+R) — charge asymmetry between Lys and Arg patches
  - fcr: Fraction of Charged Residues (K+R+D+E) / L
  - log_length: log10(L)

All are per-protein scalars. SCD and kappa encode the *patterning* of charges,
which is known to differentiate PS-capable IDRs from non-PS disordered regions.
"""

import numpy as np

# Charge states at pH ~7.4
POSITIVE = {"K", "R"}
NEGATIVE = {"D", "E"}
AROMATIC = {"F", "Y", "W"}
CHARGED = POSITIVE | NEGATIVE


def compute_sequence_features(sequence: str) -> dict:
    """Compute 6 biophysical features from an amino acid sequence.

    Args:
        sequence: protein sequence string (uppercase)

    Returns:
        dict with keys: aromatic_fraction, net_charge, scd, kappa,
                        fcr, log_length
    """
    L = len(sequence)
    if L == 0:
        return {
            "aromatic_fraction": 0.0, "net_charge": 0.0, "scd": 0.0,
            "kappa": 0.0, "fcr": 0.0, "log_length": 0.0,
        }

    seq = sequence.upper()
    counts = {aa: seq.count(aa) for aa in set(seq)}

    n_K = counts.get("K", 0)
    n_R = counts.get("R", 0)
    n_D = counts.get("D", 0)
    n_E = counts.get("E", 0)
    n_F = counts.get("F", 0)
    n_Y = counts.get("Y", 0)
    n_W = counts.get("W", 0)

    aromatic = (n_F + n_Y + n_W) / L
    n_positive = n_K + n_R
    n_negative = n_D + n_E
    net = (n_positive - n_negative) / L
    charged_count = n_positive + n_negative
    fcr = charged_count / L

    kappa = (n_K - n_R) / (n_K + n_R) if (n_K + n_R) > 0 else 0.0

    # SCD: (1/L) * sum_{i,j} q_i * q_j * sqrt(|i-j|)
    charge_map = {"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0}
    q = np.array([charge_map.get(aa, 0.0) for aa in seq], dtype=np.float32)
    if np.abs(q).sum() == 0:
        scd = 0.0
    else:
        indices = np.arange(L, dtype=np.float32)
        qi = q[:, None]
        qj = q[None, :]
        dist = np.abs(indices[:, None] - indices[None, :])
        scd_matrix = qi * qj * np.sqrt(dist)
        scd = float(scd_matrix.sum() / L)

    log_len = np.log10(max(L, 1))

    return {
        "aromatic_fraction": round(float(aromatic), 6),
        "net_charge": round(float(net), 6),
        "scd": round(scd, 6),
        "kappa": round(float(kappa), 6),
        "fcr": round(float(fcr), 6),
        "log_length": round(float(log_len), 6),
    }


def features_to_tensor(feats: dict) -> "torch.Tensor":
    """Convert feature dict to a (6,) float tensor."""
    import torch
    order = ["aromatic_fraction", "net_charge", "scd", "kappa", "fcr", "log_length"]
    return torch.tensor([feats[k] for k in order], dtype=torch.float32)


FEATURE_DIM = 6
FEATURE_NAMES = ["aromatic_fraction", "net_charge", "scd", "kappa", "fcr", "log_length"]
