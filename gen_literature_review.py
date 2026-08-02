"""Generate literature review Excel for PhaseGen paper."""
import csv, os

output = os.path.join(os.path.dirname(__file__), 'literature_review.csv')

rows = [
    ['Method', 'Year', 'Journal', 'Architecture', 'Input Features',
     'PLM Used', 'Task1 Binary', 'Task2 Self/Part', 'Trained on PPMC',
     'PPMC AUROC', 'PPMC F1', 'PPMC MCC', 'Cross-dataset', 'Key Innovation', 'Code'],

    # PhaseGen (ours)
    ['PhaseGen (ours)', '2026', 'Brief. Bioinform.',
     'ProtT5+CTN+Performer+StickerAttention+MLP',
     'ProtT5 embed + 6 biophysical (aromatic,charge,SCD,kappa,FCR,logL)',
     'ProtT5-XL (frozen)', 'Yes (608-256-2)', 'Yes (608-256-3)',
     'Yes (2300/288/288)', '0.965', '0.936', '0.843',
     'Yes (PhaSePred 12301, zero-shot 0.643, DANN 0.834)',
     'StickerAttention pooling; first systematic cross-dataset evaluation; 3D distribution shift analysis', 'Yes'],

    # PCTPS (our prior work)
    ['PCTPS (prior)', '2026', 'J. Mol. Graph. Model.',
     'ProtT5+CTN+KAN head',
     'ProtT5 embed',
     'ProtT5-XL (frozen)', 'Yes', 'No',
     'Yes', '-', '-', '-',
     'No', 'CTN-KAN framework for PS prediction', 'Yes'],

    # PPMC-benchmarked: full metrics available
    ['DeePhase', '2021', 'PNAS',
     'Gradient-boosted trees + sequence grammar',
     'Sequence composition, patterning, evolutionary conservation',
     'None', 'Yes', 'No',
     'No (evaluated)', '0.86', '0.66', '0.53',
     'No', 'Molecular grammar of condensates from sequence determinants', 'Webserver'],

    ['FuzDrop', '2020', 'Nucleic Acids Res.',
     'Sequence predictor + AlphaFold structures',
     'Sequence composition, disorder, 2ndary structure',
     'None', 'Yes', 'No',
     'No (evaluated)', '0.82', '0.61', '0.48',
     'No', 'AlphaFold-integrated interaction prediction', 'Webserver'],

    ['PSPredictor', '2022', 'BMC Bioinformatics',
     'Classical ML (SVM/RF) + handcrafted features',
     'AAC, DPC, CTD, disorder, PSSM',
     'None', 'Yes', 'No',
     'No (evaluated)', '0.81', '0.60', '0.47',
     'No', 'ML with comprehensive sequence features', 'Yes'],

    ['PSPHunter', '2024', 'Nature Communications',
     'Random Forest + multimodal features',
     'Seq embed, evolutionary, structure, disorder, PTM, PPI',
     'None', 'Yes', 'Residue-level',
     'No (evaluated)', '0.79', '0.57', '0.41',
     'No', 'Key residue identification via ML; residue-level explainability', 'Yes'],

    # PPMC-evaluated: no full metrics reported
    ['catGRANULE 2.0', '2025', 'Genome Biology',
     'Physicochemical + AlphaFold structural',
     'Physicochemical properties, AlphaFold structures',
     'None', 'Yes', 'Single-residue',
     'No (evaluated)', '-', '-', '-',
     'No', 'Single AA resolution; mutation effect evaluation', 'Yes'],

    ['LLPhyScore', '2022', 'J. Mol. Biol.',
     'Physicochemical scoring function',
     'AA physicochemical properties',
     'None', 'Yes', 'No',
     'No (evaluated)', '-', '-', '-',
     'No', 'Physics-based scoring without ML training', 'Yes'],

    ['PSAP', '2023', 'bioRxiv',
     'Classical ML classifier',
     'Sequence + evolutionary features',
     'None', 'Yes', 'No',
     'No (evaluated)', '-', '-', '-',
     'No', 'Ensemble of sequence and evolutionary features', 'Yes'],

    ['Molphase', '-', '-',
     'Physicochemical + structure',
     'Structure-based features',
     'None', 'Yes', 'No',
     'No (evaluated)', '-', '-', '-',
     'No', 'Structure-informed physicochemical prediction', '-'],

    ['PSPire', '-', '-',
     'Structure-informed DL',
     'AlphaFold2 structural features',
     'None', 'Yes', 'No',
     'No (evaluated)', '-', '-', '-',
     'No', 'Predicted structure integration', '-'],

    ['PICNIC', '-', '-',
     'Structure-informed',
     'AlphaFold2 structural features',
     'None', 'Yes', 'No',
     'No (evaluated)', '-', '-', '-',
     'No', 'Structure-based phase separation prediction', '-'],

    ['Phaseek', '2025', 'bioRxiv',
     'PLM-based DL',
     'ESM/ProtT5 embeddings',
     'Yes', 'Yes', 'No',
     'No (evaluated)', '-', '-', '-',
     'Yes (claims)', 'Claims generalizable PS prediction from sequence', '-'],

    ['Seq2Phase', '-', '-',
     'Sequence-based',
     'Sequence composition',
     'None', 'Yes', 'No',
     'No (evaluated)', '-', '-', '-',
     'No', 'Sequence composition-based predictor', '-'],

    ['PLAAC', '-', '-',
     'Prion-like domain detector',
     'Prion-like AA composition',
     'None', 'No (PLD only)', 'No',
     'No (evaluated)', '-', '-', '-',
     'No', 'Prion-like amino acid composition for LLPS propensity', '-'],

    # Newer methods (2024-2026) with two-task capability
    ['PhaseNet', '2025', 'Int. J. Biol. Macromol.',
     'ESM+CNN-BiGRU+Multi-head Attention+HSIC',
     'ESM embed + handcrafted features',
     'ESM', 'Yes', 'Yes (Lasso+Stacking ensemble)',
     'Possibly', '-', '-', '-',
     'No', 'Dual-task with HSIC regularization; separate models for two tasks', 'Yes'],

    ['Opt_PredLLPS', '2024', 'Briefings in Bioinformatics',
     'Task1: CNN+BiLSTM; Task2: XGBoost+37 features',
     'Evolutionary info + multimodal features',
     'None', 'Yes', 'Yes (XGBoost + 37 physicochemical)',
     'No (own split)', '-', '-', '-',
     'No', 'True two-stage two-task architecture; saturation mutagenesis', 'Yes'],

    ['LLPS-predict', '2026', 'PNAS',
     'PLM embedding-based classifier',
     'PLM embeddings',
     'Yes', 'Yes', 'No',
     'Not on PPMC', '-', '-', '-',
     'No', 'Proteome-wide annotation in hours; mutation sensitivity', 'Yes'],

    ['PhaSeMotif', '2026', 'Nature Communications',
     'Interpretable generative DL',
     'IDR sequences',
     'Yes', 'Yes', 'No (motif discovery)',
     'No', '-', '-', '-',
     'No', 'Interpretable+generative for IDR motifs', '-'],

    ['LLPSense', '2025', 'bioRxiv',
     'ML for condition-dependent PS',
     'Sequence + condition (pH, temp)',
     'No', 'Yes', 'No',
     'No', '-', '-', '-',
     'No', 'First to model condition-dependence (pH, temperature)', '-'],

    ['PSTP', '2024', 'bioRxiv',
     'Transfer learning + attention',
     'Sequence + attention maps',
     'No', 'Yes', 'No',
     'No', '-', '-', '-',
     'No', 'Transfer learning for latent sequence grammar', '-'],

    ['SSPSPredictor', '2026', 'bioRxiv',
     'Sequence+structure dual-channel DL',
     'Sequence + predicted structure',
     'No', 'Yes', 'No',
     'No', '-', '-', '-',
     'No', 'Dual-channel sequence and structure DL', '-'],

    ['PhosLLPS', '2025', 'Communications Chemistry',
     'DL for PTM-regulated LLPS',
     'Sequence + PTM features',
     'No', 'Yes', 'No',
     'No', '-', '-', '-',
     'No', 'First DL model for PTM regulation of PS', '-'],

    ['PredLLPS_PSSM', '2023', 'Briefings in Bioinformatics',
     'DNN + PSSM evolutionary features',
     'PSSM evolutionary profiles',
     'None', 'Yes', 'No',
     'No', '-', '-', '-',
     'No', 'PSSM-based evolutionary information for PS prediction', 'Yes'],
]

with open(output, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

print(f'Saved {len(rows)-1} methods to {output}')
