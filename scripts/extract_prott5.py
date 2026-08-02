#!/usr/bin/env python
"""Extract ProtT5-XL-UniRef50 embeddings for all proteins.

ProtT5 uses a T5-based architecture with sentencepiece tokenizer.
We extract per-residue embeddings from the last encoder hidden state
in FP16, one protein at a time, to stay within 8GB VRAM.

Output format matches the existing ESM-2 embedding .pt files for
seamless integration with EmbeddingDataset.
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str,
                       default="D:/Users/朱江龙/phasegen/models/prot_t5_xl/prot_t5_xl_uniref50",
                       help="ProtT5 model (already in FP16)")
    parser.add_argument("--fasta-file", type=str,
                       default="D:/Users/朱江龙/phasegen/data/processed/sequences.fasta")
    parser.add_argument("--output-dir", type=str,
                       default="D:/Users/朱江龙/phasegen/data/prott5_embeddings")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-len", type=int, default=1024,
                       help="Max residues per protein (truncate longer)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load ProtT5
    print(f"Loading {args.model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, legacy=True, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(args.model_name, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    print(f"Model loaded. VRAM used: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    # Parse FASTA
    print(f"Reading {args.fasta_file}...")
    sequences = {}
    with open(args.fasta_file) as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_seq:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0].split("|")[0]
                current_seq = []
            else:
                current_seq.append(line.upper())
        if current_id and current_seq:
            sequences[current_id] = "".join(current_seq)

    print(f"Loaded {len(sequences)} sequences")
    max_allocated = 0

    # Extract embeddings
    entries = {}
    for uid, seq in tqdm(sequences.items(), desc="Extracting ProtT5"):
        # Truncate to max_len
        seq = seq[:args.max_len]

        # ProtT5 expects space-separated amino acids and special prefix
        # " ".join(list(seq)) puts spaces between each AA residue
        seq_spaced = " ".join(list(seq))

        # Tokenize
        ids = tokenizer(
            seq_spaced, return_tensors="pt", padding=False, truncation=True,
            max_length=args.max_len * 2 + 3  # residues + spaces + special tokens
        )
        ids = {k: v.to(device) for k, v in ids.items()}

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            outputs = model(**ids)
            # last_hidden_state: (1, T, hidden_dim=1024)
            hidden = outputs.last_hidden_state.float().cpu()

        # Remove special tokens
        # ProtT5 with " ".join format: each AA = 1 token, plus trailing </s>
        # Take positions 0:len(seq) to skip only the </s> token
        L = min(len(seq), hidden.size(1) - 1)
        embedding = hidden[0, :L]  # (L, 1024)

        # Pad to max_len if shorter
        if embedding.size(0) < args.max_len:
            pad = torch.zeros(args.max_len - embedding.size(0), embedding.size(1))
            embedding = torch.cat([embedding, pad], dim=0)

        # Save (sanitize uid for Windows — | is invalid in filenames)
        safe_uid = uid.replace("|", "_")
        torch.save({
            "embedding": embedding,
            "length": len(seq),
            "embed_dim": embedding.size(1),
        }, os.path.join(args.output_dir, f"{safe_uid}.pt"))

        entries[uid] = {"length": len(seq), "embed_dim": embedding.size(1)}
        max_allocated = max(max_allocated, torch.cuda.max_memory_allocated())

    # Save metadata
    metadata = {
        "model": args.model_name,
        "stats": {"embed_dim": embedding.size(1), "num_entries": len(entries)},
        "entries": entries,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"\nDone! {len(entries)} embeddings saved to {args.output_dir}")
    print(f"Embed dim: {embedding.size(1)}")
    print(f"Peak VRAM: {max_allocated/1e9:.1f}GB")


if __name__ == "__main__":
    main()
