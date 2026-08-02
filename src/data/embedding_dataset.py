"""Dataset for loading pre-extracted ESM-2 embeddings directly.

This bypasses the heavy PLM forward pass, enabling fast CTN+KAN training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional


class EmbeddingDataset(Dataset):
    """Dataset that loads cached ESM-2 embeddings instead of raw sequences.

    Each item is an embedding tensor (L, embed_dim) with binary/ternary labels.
    Optionally loads pre-computed IDR conformational features.
    """

    def __init__(self, embedding_dir: str, label_csv: str, split_file: str,
                 max_len: int = 1024, idr_dir: str = None,
                 seq_feat_dir: str = None, domain_map: Dict[str, int] = None):
        """
        Args:
            embedding_dir: directory with {uniprot_id}.pt files and metadata.json
            label_csv: path to labels.csv (from preprocessing)
            split_file: path to train.txt/val.txt/test.txt (sequence IDs)
            idr_dir: optional directory with {uniprot_id}.pt IDR feature files
            seq_feat_dir: optional directory with {uniprot_id}.pt seq feature files (6,)
            domain_map: optional dict mapping uniprot_id -> domain_label (0=PPMC, 1=PhaSePred)
        """
        self.embedding_dir = embedding_dir
        self.max_len = max_len
        self.idr_dir = idr_dir
        self.seq_feat_dir = seq_feat_dir
        self.domain_map = domain_map or {}

        # Load metadata
        with open(os.path.join(embedding_dir, "metadata.json")) as f:
            meta = json.load(f)
        self.embed_dim = meta["stats"]["embed_dim"]
        self.entries = meta["entries"]

        # Load labels
        labels_df = pd.read_csv(label_csv)
        self.label_map = {}
        for _, row in labels_df.iterrows():
            self.label_map[row['id']] = {
                'binary': int(row['binary_label']),
                'ternary': int(row['ternary_label']),
            }

        # Load split
        if os.path.exists(split_file):
            with open(split_file) as f:
                split_ids = set(line.strip() for line in f if line.strip())
        else:
            split_ids = set(labels_df['id'].values)

        # Build index: only include IDs that have both embeddings and labels
        self.ids = [uid for uid in split_ids
                    if uid in self.entries and uid in self.label_map]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict:
        uid = self.ids[idx]
        safe_uid = uid.replace("|", "_")
        data = torch.load(os.path.join(self.embedding_dir, f"{safe_uid}.pt"),
                         map_location='cpu', weights_only=True)
        embedding = data["embedding"]  # (L, D)
        length = min(data["length"], self.max_len)

        if embedding.size(0) > self.max_len:
            embedding = embedding[:self.max_len]

        labels = self.label_map[uid]
        item = {
            "embedding": embedding.float(),
            "length": torch.tensor(length, dtype=torch.long),
            "binary_label": torch.tensor(labels['binary'], dtype=torch.long),
            "ternary_label": torch.tensor(labels['ternary'], dtype=torch.long),
        }

        if self.idr_dir is not None:
            idr_path = os.path.join(self.idr_dir, f"{safe_uid}.pt")
            if os.path.exists(idr_path):
                idr_feat = torch.load(idr_path, map_location='cpu', weights_only=True)
                L = embedding.size(0)
                if idr_feat.size(0) > L:
                    idr_feat = idr_feat[:L]
                elif idr_feat.size(0) < L:
                    pad = torch.zeros(L - idr_feat.size(0), idr_feat.size(1))
                    idr_feat = torch.cat([idr_feat, pad], dim=0)
            else:
                idr_feat = torch.zeros(embedding.size(0), 5)
            item["idr_features"] = idr_feat.float()

        if self.seq_feat_dir is not None:
            seq_path = os.path.join(self.seq_feat_dir, f"{safe_uid}.pt")
            if os.path.exists(seq_path):
                item["seq_features"] = torch.load(seq_path, map_location='cpu', weights_only=True).float()
            else:
                item["seq_features"] = torch.zeros(6)

        if self.domain_map:
            item["domain_label"] = torch.tensor(
                self.domain_map.get(uid, 0), dtype=torch.long)
        else:
            item["domain_label"] = torch.tensor(0, dtype=torch.long)

        return item


def embedding_collate_fn(batch: List[Dict]) -> Dict:
    """Pad variable-length embeddings to max length in batch."""
    max_len = max(item["embedding"].size(0) for item in batch)
    embed_dim = batch[0]["embedding"].size(1)

    padded = torch.zeros(len(batch), max_len, embed_dim)
    mask = torch.zeros(len(batch), max_len)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    binary_labels = torch.zeros(len(batch), dtype=torch.long)
    ternary_labels = torch.zeros(len(batch), dtype=torch.long)

    has_idr = "idr_features" in batch[0]

    for i, item in enumerate(batch):
        L = item["embedding"].size(0)
        padded[i, :L] = item["embedding"]
        mask[i, :L] = 1.0
        lengths[i] = item["length"]  # actual protein length, not padded embedding size
        binary_labels[i] = item["binary_label"]
        ternary_labels[i] = item["ternary_label"]

    result = {
        "embedding": padded,
        "mask": mask,
        "lengths": lengths,
        "binary_label": binary_labels,
        "ternary_label": ternary_labels,
    }

    if has_idr:
        D = batch[0]["idr_features"].size(-1)
        idr_padded = torch.zeros(len(batch), max_len, D)
        for i, item in enumerate(batch):
            L = min(item["idr_features"].size(0), max_len)
            idr_padded[i, :L] = item["idr_features"][:L]
        result["idr_features"] = idr_padded

    has_seq = "seq_features" in batch[0]
    if has_seq:
        result["seq_features"] = torch.stack([item["seq_features"] for item in batch])

    has_domain = "domain_label" in batch[0]
    if has_domain:
        result["domain_label"] = torch.tensor(
            [item["domain_label"].item() for item in batch], dtype=torch.long)

    return result


def create_embedding_dataloaders(embedding_dir: str, label_csv: str,
                                  splits_dir: str, batch_size: int = 32,
                                  num_workers: int = 0,
                                  idr_dir: str = None,
                                  seq_feat_dir: str = None,
                                  use_weighted_sampler: bool = False,
                                  max_len: int = 1024,
                                  domain_map_path: str = None,
                                  ) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders from cached embeddings."""
    domain_map = None
    if domain_map_path and os.path.exists(domain_map_path):
        with open(domain_map_path) as f:
            domain_map = json.load(f)

    loaders = {}
    for name in ["train", "val", "test"]:
        split_file = os.path.join(splits_dir, f"{name}.txt")
        if os.path.exists(split_file):
            dataset = EmbeddingDataset(embedding_dir, label_csv, split_file,
                                      idr_dir=idr_dir, seq_feat_dir=seq_feat_dir,
                                      domain_map=domain_map, max_len=max_len)
            if name == "train" and use_weighted_sampler:
                binary_labels = np.array([
                    int(dataset.label_map[uid]['binary']) for uid in dataset.ids
                ], dtype=np.int64)
                class_counts = np.bincount(binary_labels)
                weights = 1.0 / class_counts[binary_labels]
                sampler = WeightedRandomSampler(
                    weights, len(weights), replacement=True,
                )
                loaders[name] = DataLoader(
                    dataset, batch_size=batch_size,
                    sampler=sampler,
                    collate_fn=embedding_collate_fn,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                loaders[name] = DataLoader(
                    dataset, batch_size=batch_size,
                    shuffle=(name == "train"),
                    collate_fn=embedding_collate_fn,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=(name == "train"),
                )
    return loaders
