# PhaseGen

Deep learning model with StickerAttention pooling for phase-separating protein prediction.

**Paper:** *PhaseGen: A Deep Learning Model with Sticker-Attention Pooling for Phase-Separating Protein Prediction Reveals Cross-Dataset Generalization Barriers*  
**Authors:** Jianglong Zhu, Taigang Liu  
**Venue:** Briefings in Bioinformatics (2026)  
**Contact:** jianglong_0624@163.com, tgliu@shou.edu.cn

---

## Overview

PhaseGen is a 32.6M-parameter deep learning model that predicts phase-separating (PS) proteins from amino acid sequences. It combines:

- **ProtT5-XL** (frozen, 3B params) — per-residue protein language model embeddings
- **Multi-kernel Conv1D** (k=3,5,7,9; 256 filters each) — multi-scale local motif extraction
- **Performer Encoder** (2 layers, 8 heads, FAVOR+ linear attention)
- **StickerAttention Pooling** — learned per-residue importance, biologically motivated by the sticker-and-spacer model of LLPS
- **6 biophysical features** — aromatic fraction, net charge, SCD, Lys-Arg asymmetry, FCR, log-length
- **Multi-task training** — focal loss + ternary cross-entropy + contrastive learning
- **Stochastic Weight Averaging (SWA)**

## Key Results

| Training Strategy | PPMC AUROC | PhaSePred AUROC |
|------------------|-----------|-----------------|
| PPMC-only | 0.966 | 0.645 |
| Combined-ERM (3-seed ensemble) | 0.724 ± 0.087 | 0.788 ± 0.043 |
| Combined-DANN (3-seed ensemble) | 0.845 ± 0.025 | 0.799 ± 0.019 |

**Main finding:** Models achieving near-perfect AUROC on the PPMC benchmark collapse to near-random performance on the independent PhaSePred benchmark. Cross-dataset generalization, not single-benchmark performance, should be the standard for evaluating PS predictors.

## Quick Start

### Installation

```bash
git clone https://github.com/jiangchen0624/PhaseGen.git
cd PhaseGen
pip install -r requirements.txt
```

### One-click training (your own data)

```bash
python run_pipeline.py \
    --input data/my_data/sequences.fasta \
    --labels data/my_data/labels.csv \
    --name my_experiment
```

## Reproducing Paper Results

### Option A: From pre-computed data (recommended)

1. **Download pre-computed assets** from Zenodo/Figshare (link to be added upon publication):

   Extract the archive into the project root:
   ```
   PhaseGen/
   ├── data/prott5_embeddings/       # ProtT5-XL embeddings for PPMC
   ├── data/llps-benchmark/          # PPMC dataset (sequences, labels, splits, features)
   ├── data/phasepred_benchmark/     # PhaSePred dataset (train+test, embeddings, labels)
   ├── data/combined/                # Combined PPMC+PhaSePred (no test leakage)
   └── checkpoints/                  # Trained model weights
   ```

2. **Evaluate pre-trained checkpoints:**
   ```bash
   # PPMC-only model on both datasets
   python scripts/evaluate.py --checkpoint checkpoints/llps_v2_full/best_model.pt

   # Combined-ERM model
   python scripts/evaluate.py --checkpoint checkpoints/combined_erm_seed42/best_model.pt

   # Combined-DANN model
   python scripts/evaluate.py --checkpoint checkpoints/combined_dann_seed42/best_model.pt
   ```

### Option B: From scratch (full pipeline)

If you want to reproduce everything from raw sequences:

**Prerequisites:** GPU with ≥8GB VRAM, ~100GB disk space for ProtT5-XL and embeddings.

#### 1. Download ProtT5-XL

The model is downloaded automatically by `transformers` on first use, or you can pre-download:
```bash
python -c "from transformers import T5EncoderModel, T5Tokenizer; \
    T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50'); \
    T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')"
```

#### 2. Prepare PPMC dataset

Download the PPMC benchmark from [Pintado-Grima et al. (2025)](https://github.com/cpintado/PPMC) and place files under `data/llps-benchmark/`.

#### 3. Prepare PhaSePred dataset

Download PhaSePred from [Chen et al. (2022)](https://github.com/chenzhongxiang/PhaSePred) and place files under `data/phasepred_benchmark/`.

#### 4. Extract ProtT5 embeddings

```bash
# PPMC proteins (~2,876 proteins, ~2 hours on RTX 4060)
python scripts/extract_prott5.py \
    --fasta data/llps-benchmark/sequences.fasta \
    --labels data/llps-benchmark/labels.csv \
    --output_dir data/llps-benchmark/prott5_embeddings

# PhaSePred training proteins (~3,342 proteins)
python scripts/extract_prott5.py \
    --fasta data/phasepred_benchmark/train_sequences.fasta \
    --labels data/phasepred_benchmark/train_labels.csv \
    --output_dir data/phasepred_benchmark/prott5_train_embeddings

# PhaSePred test proteins (~12,301 proteins)
python scripts/extract_prott5.py \
    --fasta data/phasepred_benchmark/test_sequences.fasta \
    --labels data/phasepred_benchmark/test_labels.csv \
    --output_dir data/phasepred_benchmark/prott5_test_embeddings
```

#### 5. Compute biophysical features

```bash
python scripts/precompute_seq_features.py \
    --fasta data/llps-benchmark/sequences.fasta \
    --output_dir data/llps-benchmark/seq_features \
    --split_dir data/llps-benchmark/splits

python scripts/precompute_seq_features.py \
    --fasta data/phasepred_benchmark/test_sequences.fasta \
    --output_dir data/phasepred_benchmark/seq_features \
    --split_dir data/phasepred_benchmark/splits
```

#### 6. Create data splits (CD-HIT at 40% identity)

```bash
python scripts/create_splits.py \
    --labels data/llps-benchmark/labels.csv \
    --output_dir data/llps-benchmark/splits
```

#### 7. Train models

```bash
# PPMC-only (3 seeds)
for seed in 42 123 456; do
    python scripts/train.py \
        --embedding_dir data/llps-benchmark/prott5_embeddings \
        --label_csv data/llps-benchmark/labels.csv \
        --splits_dir data/llps-benchmark/splits \
        --seq_feat_dir data/llps-benchmark/seq_features \
        --output_dir checkpoints/llps_v3_seed${seed} \
        --epochs 30 --batch_size 32 --use_sticker_pooling --use_swa \
        --performer_d_model 1024 --performer_n_layers 2 \
        --seed ${seed}
done

# Combined-ERM (3 seeds)
for seed in 42 123 456; do
    python scripts/train.py \
        --embedding_dir data/combined/embeddings \
        --label_csv data/combined/labels.csv \
        --splits_dir data/combined/splits \
        --seq_feat_dir data/combined/seq_features \
        --output_dir checkpoints/combined_erm_seed${seed} \
        --epochs 30 --batch_size 32 --use_sticker_pooling --use_swa \
        --performer_d_model 1024 --performer_n_layers 2 \
        --seed ${seed}
done

# Combined-DANN (3 seeds)
for seed in 42 123 456; do
    python scripts/train.py \
        --embedding_dir data/combined/embeddings \
        --label_csv data/combined/labels.csv \
        --splits_dir data/combined/splits \
        --seq_feat_dir data/combined/seq_features \
        --output_dir checkpoints/combined_dann_seed${seed} \
        --epochs 30 --batch_size 32 --use_sticker_pooling --use_swa \
        --performer_d_model 1024 --performer_n_layers 2 \
        --domain_adapt adversarial \
        --domain_map_path data/combined/domain_map.json \
        --seed ${seed}
done
```

#### 8. Evaluate and ensemble

```bash
# Single-model evaluation
python scripts/evaluate.py --checkpoint checkpoints/llps_v3_seed42/best_model.pt

# Cross-dataset evaluation (evaluates on both PPMC and PhaSePred test sets)
python scripts/evaluate.py --checkpoint checkpoints/llps_v3_seed42/best_model.pt

# Ensemble predictions (average probabilities across 3 seeds)
python scripts/evaluate.py --ensemble checkpoints/llps_v3_seed42,checkpoints/llps_v3_seed123,checkpoints/llps_v3_seed456
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
- `binary_label`: 0 = non-PS, 1 = PS
- `ternary_label`: 0 = non-PS, 1 = PS-Self, 2 = PS-Part

## Architecture

```
ProtT5-XL (frozen, 3B params)
  → Multi-kernel Conv1D (k=3,5,7,9; 256 filters each)
  → Performer Encoder (2 layers, 8 heads, 256 random features)
  → StickerAttention Pooling
  → Concatenate with 6-d biophysical features
  → MLP Classification Head (512→256→2 binary, 512→256→3 ternary)
  → SWA (stochastic weight averaging)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- transformers >= 4.30.0
- ProtT5-XL (~12GB disk space)

See `requirements.txt` for full list.

## Citation

```bibtex
@article{zhu2026phasegen,
  title={PhaseGen: A Deep Learning Model with Sticker-Attention Pooling for
         Phase-Separating Protein Prediction Reveals Cross-Dataset Generalization Barriers},
  author={Zhu, Jianglong and Liu, Taigang},
  journal={Briefings in Bioinformatics},
  year={2026},
  note={In press}
}
```

## License

MIT License. See `LICENSE` for details.