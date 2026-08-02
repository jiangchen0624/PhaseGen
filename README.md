# PhaseGen

Deep learning model with StickerAttention pooling for phase-separating protein prediction.

Paper: *PhaseGen: A Deep Learning Model with Sticker-Attention Pooling for Phase-Separating Protein Prediction Reveals Cross-Dataset Generalization Barriers* (Briefings in Bioinformatics)

## Quick Start

```bash
# Install
pip install -r requirements.txt

# One-click training
python run_pipeline.py --input data/ppmc/sequences.fasta --labels data/ppmc/labels.csv --name my_exp
```

## Prepare Your Own Dataset

Two files needed under `data/<dataset_name>/`:

**sequences.fasta** — Standard FASTA format:
```
>P12345
MADSKLVG...
>Q67890
KLEGHTY...
```

**labels.csv** — Three columns:
```csv
id,binary_label,ternary_label
P12345,1,1
Q67890,0,0
```
- `binary_label`: 0=non-PS, 1=PS
- `ternary_label`: 0=non-PS, 1=PS-Self, 2=PS-Part

## Manual Steps (if not using pipeline)

```bash
# 1. Extract ProtT5 embeddings
python scripts/extract_prott5.py --fasta data/my_data/sequences.fasta --labels data/my_data/labels.csv --output_dir data/my_data/prott5_embeddings

# 2. Compute biophysical features
python scripts/precompute_seq_features.py --fasta data/my_data/sequences.fasta --output_dir data/my_data/seq_features

# 3. Create splits
python scripts/create_splits.py --labels data/my_data/labels.csv --output_dir data/my_data/splits

# 4. Train
python scripts/train.py --embedding_dir data/my_data/prott5_embeddings --label_csv data/my_data/labels.csv --splits_dir data/my_data/splits --seq_feat_dir data/my_data/seq_features --output_dir checkpoints/my_exp --epochs 30
```

## Evaluate

```bash
python scripts/evaluate.py --checkpoint_dir checkpoints/my_exp
```

## Project Structure

```
PhaseGen_1/
├── run_pipeline.py          # One-click training entry point
├── requirements.txt
├── scripts/
│   ├── train.py             # Model training (LightweightModel = CTN + Performer + StickerAttention)
│   ├── evaluate.py          # Reproduce paper metrics from saved checkpoints
│   ├── extract_prott5.py    # Extract ProtT5-XL embeddings
│   ├── precompute_seq_features.py  # Compute 6 biophysical features
│   └── create_splits.py     # Train/val/test split
├── src/
│   ├── models/              # CTN, Performer, StickerAttention, DANN
│   ├── losses/              # Focal loss, contrastive loss
│   ├── data/                # Embedding dataset loader
│   ├── training/            # Evaluator, scheduler
│   └── utils/               # Sequence features, seed
└── data/                    # Your datasets go here
```
