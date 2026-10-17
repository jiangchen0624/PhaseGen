#!/usr/bin/env python
"""Fast training on pre-extracted protein language model embeddings.

PhaseGen uses ProtT5-XL embeddings (1024-dim) as frozen input. This training
script loads pre-computed .pt embedding files and trains the downstream
CTN + Performer + StickerAttention + MLP architecture.

See README.md for the full data preparation pipeline (extract embeddings →
compute features → create splits → train).

Usage (after data preparation):
    python scripts/train.py --embedding_dir data/my_data/prott5_embeddings \\
        --label_csv data/my_data/labels.csv --splits_dir data/my_data/splits \\
        --seq_feat_dir data/my_data/seq_features --output_dir checkpoints/my_exp \\
        --epochs 30 --use_sticker_pooling --use_swa
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from src.models.ctn import CTN
from src.models.kan import KANHead
from src.models.idr_embedder import IDREmbedder
from src.models.sticker_attention import SeqFeatureEmbedder
from src.models.domain_discriminator import DomainAdversarialModule
from src.losses.multi_task_loss import MultiTaskLoss
from src.training.evaluator import compute_metrics, format_metrics
from src.training.scheduler import CosineWarmupScheduler
from src.data.embedding_dataset import create_embedding_dataloaders, EmbeddingDataset
from src.utils.seed import set_seed
from src.utils.sequence_features import FEATURE_DIM


class EMAModel:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup.clear()


class LightweightModel(nn.Module):
    """CTN + sticker pooling + biophysical features + classification heads.

    Architecture:
        ProtT5 embedding → CTN (multi-scale conv + Performer) → StickerAttentionPooling
        Biophysical features → SeqFeatureEmbedder
        [CTN output; SeqFeat embedding] → concat → MLP heads

    Key components:
    - StickerAttentionPooling: biologically-motivated pooling that learns to attend
      to sticker residues (F/Y/W) from ProtT5 embeddings, producing interpretable
      per-residue importance scores.
    - SeqFeatureEmbedder: embeds 6 scalar biophysical features (aromatic fraction,
      net charge, SCD, kappa, FCR, length) into a dense vector concatenated
      before the classification head. These features directly encode the
      sticker-spacer framework physics.
    """

    def __init__(self, embed_dim=1280, conv_filters=256, conv_kernels=None,
                 performer_d_model=1024, performer_n_heads=8, performer_n_layers=2,
                 performer_nb_features=256, output_dim=512,
                 kan_hidden=256, kan_grid=5, kan_order=3, kan_dropout=0.1,
                 use_performer=True, use_mamba=False,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 use_kan=False,
                 use_attention_pooling=False, pooling_n_queries=4, pooling_n_heads=8,
                 use_sticker_pooling=False,
                 use_idr=False, use_global_idr=True,
                 idr_property_dim=66, idr_global_dim=128, idr_dropout=0.1,
                 use_seq_features=False, seq_feat_dim=96,
                 performer_dropout=0.1, conv_dropout=0.2,
                 domain_adapt="none", domain_lambda_max=1.0,
                 domain_warmup_steps=1000):
        super().__init__()

        self.use_idr = use_idr
        self.use_global_idr = use_global_idr and use_idr
        self.use_attention_pooling = use_attention_pooling
        self.use_sticker_pooling = use_sticker_pooling
        self.use_seq_features = use_seq_features
        self.domain_adapt = domain_adapt

        if use_idr:
            mode = "global" if self.use_global_idr else "per_residue"
            self.idr_embedder = IDREmbedder(
                property_dim=idr_property_dim, dropout=idr_dropout,
                mode=mode, global_out_dim=idr_global_dim,
            )
            if self.use_global_idr:
                ctn_in_dim = embed_dim
                head_in_dim = output_dim + idr_global_dim
            else:
                ctn_in_dim = embed_dim + self.idr_embedder.out_dim
                head_in_dim = output_dim
        else:
            self.idr_embedder = None
            ctn_in_dim = embed_dim
            head_in_dim = output_dim

        # Biophysical sequence features: 6 scalars → embedded vector
        if use_seq_features:
            self.seq_embedder = SeqFeatureEmbedder(
                n_features=FEATURE_DIM, output_dim=seq_feat_dim,
                dropout=kan_dropout,
            )
            head_in_dim = head_in_dim + seq_feat_dim
        else:
            self.seq_embedder = None

        self.ctn = CTN(
            in_channels=ctn_in_dim, conv_filters=conv_filters,
            conv_kernel_sizes=conv_kernels or [3, 5, 7, 9],
            performer_d_model=performer_d_model,
            performer_n_heads=performer_n_heads,
            performer_n_layers=performer_n_layers,
            performer_nb_features=performer_nb_features,
            output_dim=output_dim,
            use_performer=use_performer,
            use_mamba=use_mamba,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            use_attention_pooling=use_attention_pooling,
            pooling_n_queries=pooling_n_queries,
            pooling_n_heads=pooling_n_heads,
            use_sticker_pooling=use_sticker_pooling,
            performer_dropout=performer_dropout,
            conv_dropout=conv_dropout,
        )

        if use_kan:
            self.binary_head = KANHead(
                in_dim=head_in_dim, hidden_dim=kan_hidden, out_dim=2,
                grid_size=kan_grid, spline_order=kan_order, dropout=kan_dropout,
            )
            self.ternary_head = KANHead(
                in_dim=head_in_dim, hidden_dim=kan_hidden, out_dim=3,
                grid_size=kan_grid, spline_order=kan_order, dropout=kan_dropout,
            )
        else:
            self.binary_head = nn.Sequential(
                nn.Linear(head_in_dim, kan_hidden), nn.LayerNorm(kan_hidden),
                nn.GELU(), nn.Dropout(kan_dropout),
                nn.Linear(kan_hidden, 2),
            )
            self.ternary_head = nn.Sequential(
                nn.Linear(head_in_dim, kan_hidden), nn.LayerNorm(kan_hidden),
                nn.GELU(), nn.Dropout(kan_dropout),
                nn.Linear(kan_hidden, 3),
            )

        self.projection_head = nn.Sequential(
            nn.Linear(head_in_dim, 256), nn.GELU(), nn.Linear(256, 128),
        )

        self.domain_adversarial = None
        if domain_adapt == "adversarial":
            self.domain_adversarial = DomainAdversarialModule(
                input_dim=output_dim,
                lambda_max=domain_lambda_max,
                warmup_steps=domain_warmup_steps,
            )

    def forward(self, embeddings, mask=None, idr_features=None, seq_features=None):
        shared, attn_weights, ctn_features = None, None, None

        if self.use_global_idr and idr_features is not None:
            idr_global = self.idr_embedder(idr_features)
            shared, attn_weights = self.ctn(embeddings, mask)
            ctn_features = shared
            shared = torch.cat([shared, idr_global], dim=-1)
        elif self.use_idr and idr_features is not None:
            idr_emb = self.idr_embedder(idr_features)
            embeddings = torch.cat([embeddings, idr_emb], dim=-1)
            shared, attn_weights = self.ctn(embeddings, mask)
            ctn_features = shared
        else:
            shared, attn_weights = self.ctn(embeddings, mask)
            ctn_features = shared

        if self.use_seq_features and seq_features is not None:
            seq_emb = self.seq_embedder(seq_features)
            shared = torch.cat([shared, seq_emb], dim=-1)

        result = {
            "binary_logits": self.binary_head(shared),
            "ternary_logits": self.ternary_head(shared),
            "projection": nn.functional.normalize(self.projection_head(shared), dim=-1),
            "attn_weights": attn_weights,
        }
        if self.domain_adapt == "adversarial" and ctn_features is not None:
            result["ctn_features"] = ctn_features
        return result




def train_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device,
                grad_accum, max_grad_norm, use_amp, ema=None,
                domain_loss_weight=0.1):
    model.train()
    total_loss = 0.0
    total_domain_loss = 0.0
    all_bin_logits, all_bin_labels = [], []
    all_domain_correct, all_domain_total = 0, 0

    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        emb = batch["embedding"].to(device)
        mask = batch["mask"].to(device)
        bin_labels = batch["binary_label"].to(device)
        ter_labels = batch["ternary_label"].to(device)
        domain_labels = batch.get("domain_label")
        if domain_labels is not None:
            domain_labels = domain_labels.to(device)
        idr = batch.get("idr_features")
        if idr is not None:
            idr = idr.to(device)
        seq_feat = batch.get("seq_features")
        if seq_feat is not None:
            seq_feat = seq_feat.to(device)

        with autocast(device_type='cuda', enabled=use_amp):
            outputs = model(emb, mask, idr_features=idr, seq_features=seq_feat)
            losses = loss_fn(outputs, bin_labels, ter_labels)

            if (model.domain_adversarial is not None and
                    "ctn_features" in outputs and domain_labels is not None):
                domain_result = model.domain_adversarial(
                    outputs["ctn_features"], domain_labels)
                losses["domain"] = domain_result["loss"]
                losses["total"] = losses["total"] + domain_loss_weight * losses["domain"]
                all_domain_correct += int(
                    domain_result["accuracy"] * len(domain_labels))
                all_domain_total += len(domain_labels)

            loss = losses["total"] / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            if ema is not None:
                ema.update()

        total_loss += losses["total"].item()
        if "domain" in losses:
            total_domain_loss += losses["domain"].item()
        all_bin_logits.append(outputs["binary_logits"].detach())
        all_bin_labels.append(bin_labels.detach())

    bin_logits = torch.cat(all_bin_logits)
    bin_labels = torch.cat(all_bin_labels)
    nan_mask = torch.isnan(bin_logits).any(dim=-1)
    if nan_mask.any():
        bin_logits = bin_logits[~nan_mask]
        bin_labels = bin_labels[~nan_mask]
    metrics = compute_metrics(bin_logits, bin_labels)
    metrics["loss"] = total_loss / len(loader)
    if total_domain_loss > 0:
        metrics["domain_loss"] = total_domain_loss / len(loader)
    if all_domain_total > 0:
        metrics["domain_accuracy"] = all_domain_correct / all_domain_total
    return metrics


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp, track_domains=False):
    model.eval()
    total_loss = 0.0
    all_bin_logits, all_bin_labels = [], []
    all_ter_logits, all_ter_labels = [], []
    all_domain_labels = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        emb = batch["embedding"].to(device)
        mask = batch["mask"].to(device)
        bin_labels = batch["binary_label"].to(device)
        ter_labels = batch["ternary_label"].to(device)
        domain_labels = batch.get("domain_label")
        if domain_labels is not None:
            all_domain_labels.append(domain_labels)
        idr = batch.get("idr_features")
        if idr is not None:
            idr = idr.to(device)
        seq_feat = batch.get("seq_features")
        if seq_feat is not None:
            seq_feat = seq_feat.to(device)

        with autocast(device_type='cuda', enabled=use_amp):
            outputs = model(emb, mask, idr_features=idr, seq_features=seq_feat)
            losses = loss_fn(outputs, bin_labels, ter_labels)

        total_loss += losses["total"].item()
        all_bin_logits.append(outputs["binary_logits"])
        all_bin_labels.append(bin_labels)
        all_ter_logits.append(outputs["ternary_logits"])
        all_ter_labels.append(ter_labels)

    if not all_bin_logits:
        return {"binary_auroc": 0.5, "binary_auprc": 0.0, "binary_accuracy": 0.0,
                "binary_f1": 0.0, "binary_mcc": 0.0, "loss": 0.0}
    bin_logits = torch.cat(all_bin_logits)
    bin_labels = torch.cat(all_bin_labels)
    ter_logits = torch.cat(all_ter_logits) if all_ter_logits else None
    ter_labels = torch.cat(all_ter_labels) if all_ter_labels else None
    nan_mask = torch.isnan(bin_logits).any(dim=-1)
    if nan_mask.any():
        bin_logits = bin_logits[~nan_mask]
        bin_labels = bin_labels[~nan_mask]
        if ter_logits is not None:
            ter_logits = ter_logits[~nan_mask]
            ter_labels = ter_labels[~nan_mask]
            if all_domain_labels:
                all_domain_labels = [dl[~nan_mask.cpu()] for dl in all_domain_labels]

    metrics = compute_metrics(bin_logits, bin_labels, ter_logits, ter_labels)
    metrics["loss"] = total_loss / max(len(loader), 1)

    # Per-domain metrics
    if track_domains and all_domain_labels:
        domain_labels = torch.cat([dl.cpu() for dl in all_domain_labels])
        for domain_id, domain_name in [(0, "PPMC"), (1, "PhaSePred")]:
            mask = domain_labels == domain_id
            if mask.sum() > 1:
                domain_metrics = compute_metrics(
                    bin_logits[mask], bin_labels[mask])
                for k, v in domain_metrics.items():
                    if isinstance(v, float) and not k.startswith("ternary"):
                        metrics[f"{domain_name}_{k}"] = v

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fast PCTPS training on cached embeddings")
    parser.add_argument("--embedding-dir", type=str,
                       default="D:/Users/朱江龙/phasegen/data/embeddings")
    parser.add_argument("--idr-dir", type=str,
                       default="D:/Users/朱江龙/phasegen/data/idr_features")
    parser.add_argument("--label-csv", type=str,
                       default="D:/Users/朱江龙/phasegen/data/processed/labels.csv")
    parser.add_argument("--splits-dir", type=str,
                       default="D:/Users/朱江龙/phasegen/data/splits")
    parser.add_argument("--output-dir", type=str,
                       default="outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=1024,
                       help="Max sequence length (reduce for memory-hungry models)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-performer", action="store_true")
    parser.add_argument("--use-mamba", action="store_true",
                       help="Use Mamba SSM encoder instead of Performer")
    parser.add_argument("--mamba-d-state", type=int, default=16,
                       help="Mamba SSM state dimension")
    parser.add_argument("--mamba-d-conv", type=int, default=4,
                       help="Mamba conv1d kernel size")
    parser.add_argument("--mamba-expand", type=int, default=2,
                       help="Mamba inner dimension expansion factor")
    parser.add_argument("--performer-d-model", type=int, default=512)
    parser.add_argument("--performer-n-layers", type=int, default=1)
    parser.add_argument("--use-kan", action="store_true")
    parser.add_argument("--use-attention-pooling", action="store_true")
    parser.add_argument("--pooling-n-queries", type=int, default=4)
    parser.add_argument("--pooling-n-heads", type=int, default=8)
    parser.add_argument("--use-idr", action="store_true")
    parser.add_argument("--per-residue-idr", action="store_true",
                       help="Use legacy per-residue IDR (default: global IDR injection)")
    parser.add_argument("--idr-global-dim", type=int, default=128,
                       help="Dimension of global IDR embedding")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--use-ema", action="store_true",
                       help="Use exponential moving average of model weights")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                       help="EMA decay rate")
    # --- Sticker pooling & biophysical features ---
    parser.add_argument("--use-sticker-pooling", action="store_true",
                       help="Use sticker-aware attention pooling (biologically motivated)")
    parser.add_argument("--seq-feat-dir", type=str,
                       default="D:/Users/朱江龙/phasegen/data/llps-benchmark/seq_features",
                       help="Directory with precomputed biophysical sequence features")
    parser.add_argument("--no-seq-features", action="store_true",
                       help="Disable biophysical sequence features even if available")
    parser.add_argument("--seq-feat-dim", type=int, default=96,
                       help="Output dimension of SeqFeatureEmbedder")
    # --- SWA (Stochastic Weight Averaging) ---
    parser.add_argument("--use-swa", action="store_true",
                       help="Use stochastic weight averaging for smoother convergence")
    parser.add_argument("--swa-start", type=int, default=10,
                       help="Epoch to start collecting SWA snapshots")
    parser.add_argument("--swa-lr", type=float, default=1e-4,
                       help="Learning rate during SWA collection phase")
    # --- Two-stage curriculum ---
    parser.add_argument("--two-stage", action="store_true",
                       help="Stage1: NP vs PS only; Stage2: add ND with higher weight")
    # Loss weights
    parser.add_argument("--binary-weight", type=float, default=1.0)
    parser.add_argument("--ternary-weight", type=float, default=0.5)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temp", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-focal", action="store_true",
                       help="Use standard CrossEntropyLoss instead of Focal Loss")
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate throughout the model")
    # Domain adaptation
    parser.add_argument("--domain-adapt", type=str, default="none",
                       choices=["none", "adversarial", "importance"],
                       help="Domain adaptation strategy for multi-dataset training")
    parser.add_argument("--domain-lambda-max", type=float, default=1.0,
                       help="Max gradient reversal lambda for DANN")
    parser.add_argument("--domain-warmup-fraction", type=float, default=0.3,
                       help="Fraction of total steps for GRL lambda warmup")
    parser.add_argument("--domain-loss-weight", type=float, default=0.1,
                       help="Weight of domain adversarial loss in total loss")
    parser.add_argument("--domain-map-path", type=str, default=None,
                       help="Path to domain_labels.json for multi-dataset training")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # DataLoaders with optional IDR features and biophysical sequence features
    loaders = create_embedding_dataloaders(
        args.embedding_dir, args.label_csv, args.splits_dir,
        batch_size=args.batch_size,
        idr_dir=args.idr_dir if args.use_idr else None,
        seq_feat_dir=args.seq_feat_dir if not args.no_seq_features else None,
        use_weighted_sampler=args.use_weighted_sampler,
        max_len=args.max_len,
        domain_map_path=args.domain_map_path if args.domain_adapt != "none" else None,
    )
    print(f"Train: {len(loaders['train'].dataset)} | Val: {len(loaders['val'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")

    # Auto-detect embed_dim from dataset metadata
    embed_dim = loaders['train'].dataset.embed_dim
    print(f"Embed dim (auto-detected): {embed_dim}")

    # Compute domain warmup steps
    total_steps_est = args.epochs * len(loaders["train"])
    domain_warmup_steps = int(args.domain_warmup_fraction * total_steps_est)

    # Model
    model = LightweightModel(
        embed_dim=embed_dim,
        use_performer=not args.no_performer,
        use_mamba=args.use_mamba,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        performer_d_model=args.performer_d_model,
        performer_n_layers=args.performer_n_layers,
        use_kan=args.use_kan,
        use_attention_pooling=args.use_attention_pooling,
        pooling_n_queries=args.pooling_n_queries,
        pooling_n_heads=args.pooling_n_heads,
        use_sticker_pooling=args.use_sticker_pooling,
        use_idr=args.use_idr,
        use_global_idr=not args.per_residue_idr,
        idr_global_dim=args.idr_global_dim,
        use_seq_features=not args.no_seq_features,
        seq_feat_dim=args.seq_feat_dim,
        kan_dropout=args.dropout,
        performer_dropout=args.dropout,
        conv_dropout=args.dropout * 2,
        domain_adapt=args.domain_adapt,
        domain_lambda_max=args.domain_lambda_max,
        domain_warmup_steps=domain_warmup_steps,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    # Loss
    loss_fn = MultiTaskLoss(
        binary_weight=args.binary_weight, ternary_weight=args.ternary_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temp,
        label_smoothing=0.0, use_focal=not args.no_focal,
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
    )

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loaders["train"])
    scheduler = CosineWarmupScheduler(optimizer, args.warmup_steps, total_steps)
    scaler = GradScaler('cuda', enabled=not args.no_amp)
    use_amp = not args.no_amp and device.type == "cuda"
    # Performer + IDR + AMP => occasional NaN; disable AMP for this combo
    safe_amp = use_amp and not (not args.no_performer and args.use_idr)
    if use_amp and not safe_amp:
        print("AMP disabled for Performer+IDR to avoid NaN instability")
        scaler = GradScaler('cuda', enabled=False)

    # EMA
    ema = EMAModel(model, decay=args.ema_decay) if args.use_ema else None

    # SWA
    swa_model = None
    swa_scheduler = None
    if args.use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer, swa_lr=args.swa_lr,
        )

    # Track per-domain best
    track_domains = args.domain_adapt != "none"
    best_auroc = -1
    best_ppmc_auroc = -1
    best_phasepred_auroc = -1
    best_epoch = 0
    patience_counter = 0
    global_step = 0

    for epoch in range(args.epochs):
        loss_fn.set_epoch(epoch)

        train_metrics = train_epoch(
            model, loaders["train"], loss_fn, optimizer, scheduler,
            scaler, device, args.grad_accum, args.max_grad_norm, safe_amp,
            ema=ema, domain_loss_weight=args.domain_loss_weight,
        )

        # Step the GRL lambda after each epoch's training
        global_step += len(loaders["train"])
        if model.domain_adversarial is not None:
            model.domain_adversarial.grl.set_lambda(global_step)

        # SWA: collect model snapshots after swa_start (scheduler is stepped inside train_epoch)
        if swa_model is not None and epoch >= args.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        val_metrics = evaluate(model, loaders["val"], loss_fn, device, safe_amp,
                               track_domains=track_domains)

        train_auc = train_metrics.get("binary_auroc", 0)
        val_auc = val_metrics.get("binary_auroc", 0)
        val_f1 = val_metrics.get("binary_f1", 0)

        extra_info = ""
        if track_domains:
            ppmc_auc = val_metrics.get("PPMC_binary_auroc", 0)
            psp_auc = val_metrics.get("PhaSePred_binary_auroc", 0)
            dom_acc = train_metrics.get("domain_accuracy", 0)
            grl_lambda = (model.domain_adversarial.grl.get_lambda()
                          if model.domain_adversarial else 0)
            extra_info = f" | PPMC_AUC={ppmc_auc:.4f} PSP_AUC={psp_auc:.4f} DA={dom_acc:.2f} λ={grl_lambda:.3f}"
        if swa_model and epoch >= args.swa_start:
            extra_info += " [SWA]"

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train loss={train_metrics['loss']:.4f} AUC={train_auc:.4f} | "
              f"Val loss={val_metrics['loss']:.4f} AUC={val_auc:.4f} F1={val_f1:.4f}"
              + extra_info)

        if val_auc > best_auroc:
            best_auroc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_metrics": val_metrics, "train_metrics": train_metrics,
                        "args": vars(args)},
                       os.path.join(args.output_dir, "best_model.pt"))
        else:
            patience_counter += 1

        # Save per-domain best checkpoints
        if track_domains:
            ppmc_auc = val_metrics.get("PPMC_binary_auroc", 0)
            if ppmc_auc > best_ppmc_auroc:
                best_ppmc_auroc = ppmc_auc
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                           "val_metrics": val_metrics, "args": vars(args)},
                          os.path.join(args.output_dir, "best_model_ppmc.pt"))
            psp_auc = val_metrics.get("PhaSePred_binary_auroc", 0)
            if psp_auc > best_phasepred_auroc:
                best_phasepred_auroc = psp_auc
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                           "val_metrics": val_metrics, "args": vars(args)},
                          os.path.join(args.output_dir, "best_model_phasepred.pt"))

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Test evaluation
    print(f"\nBest epoch: {best_epoch+1} (val AUROC={best_auroc:.4f})")
    if track_domains:
        print(f"Best PPMC AUROC: {best_ppmc_auroc:.4f}  Best PhaSePred AUROC: {best_phasepred_auroc:.4f}")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=False)
    test_metrics = evaluate(model, loaders["test"], loss_fn, device, use_amp,
                           track_domains=track_domains)
    print(f"\n=== Test Results (best model) ===")
    print(format_metrics(test_metrics))
    if track_domains:
        for dname in ["PPMC", "PhaSePred"]:
            auc = test_metrics.get(f"{dname}_binary_auroc", 0)
            f1 = test_metrics.get(f"{dname}_binary_f1", 0)
            if auc > 0:
                print(f"  {dname}: AUC={auc:.4f} F1={f1:.4f}")

    # Test with EMA model if enabled
    if ema is not None:
        ema.apply_shadow()
        ema_test_metrics = evaluate(model, loaders["test"], loss_fn, device, use_amp)
        ema.restore()
        print(f"\n=== Test Results (EMA model) ===")
        print(format_metrics(ema_test_metrics))
        torch.save({"epoch": best_epoch, "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics, "test_metrics": ema_test_metrics,
                    "args": vars(args)},
                   os.path.join(args.output_dir, "best_model_ema.pt"))

    # Test with SWA model if enabled
    if swa_model is not None:
        # Note: skip update_bn since our model takes dict batch inputs
        swa_test_metrics = evaluate(swa_model.module, loaders["test"], loss_fn,
                                    device, use_amp)
        print(f"\n=== Test Results (SWA model) ===")
        print(format_metrics(swa_test_metrics))
        torch.save({"epoch": best_epoch, "model_state_dict": swa_model.module.state_dict(),
                    "val_metrics": val_metrics, "test_metrics": swa_test_metrics,
                    "args": vars(args)},
                   os.path.join(args.output_dir, "best_model_swa.pt"))
        return swa_test_metrics

    return test_metrics


if __name__ == "__main__":
    main()
