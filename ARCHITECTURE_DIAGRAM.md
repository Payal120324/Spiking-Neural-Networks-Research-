# Spike-Legal-NLP Architecture Diagram

## Overview

This is a research framework for comparing Transformer-based Legal NLP models with Spike Encoding + Spiking Neural Networks (SNNs) for legal text classification under domain shift.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SPIKE-LEGAL-NLP RESEARCH FRAMEWORK                            │
│  A Semantic- and Energy-Aware Study of Spike Encoding for Legal Text             │
│  Classification Under Domain Shift                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## High-Level Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   CLI Entry   │────▶│  Pipeline    │────▶│   Results    │
│   (main.py)   │     │  Orchestrator│     │  & Reports   │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │ Dataset  │      │  Models  │      │Encoding  │
    │ Manager  │      │          │      │ Modules  │
    └──────────┘      └──────────┘      └──────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │Evaluation│      │Visualiza- │      │Reporting │
    │  Modules │      │   tion   │      │ Generator│
    └──────────┘      └──────────┘      └──────────┘
```

## Detailed Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           EXPERIMENT PIPELINE (9 Stages)                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Dataset Loading                                                             │
│ ───────────────────────────────────────────────────────────────────────────────────  │
│                                                                                     │
│  CLI Args: --dataset case_hold                                                       │
│           ┌──────────────┐                                                          │
│           │ DatasetManager│                                                         │
│           └──────┬───────┘                                                          │
│                  │                                                                  │
│                  ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ DATASET SOURCES:                                                              │   │
│  │ • HuggingFace (LexGLUE): case_hold, ecthr_a, ecthr_b, eurlex, ledgar, etc.   │   │
│  │ • Custom: CSV, JSON, JSONL, Excel, Parquet                                   │   │
│  │                                                                               │   │
│  │ OUTPUT: {train: [{text, label}], validation: [...], test: [...]}            │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                  │                                                                  │
│                  ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ DatasetStatistics: label distribution, word counts, class balance            │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Transformer Embeddings                                                      │
│ ───────────────────────────────────────────────────────────────────────────────────  │
│                                                                                     │
│  CLI Args: --encoder legal_bert                                                       │
│           ┌─────────────────────────────────────────────────────────────────────┐  │
│           │                    TransformerBaseline                                │  │
│           │  ┌────────────────────────────────────────────────────────────────┐  │  │
│           │  │ Supported Models:                                              │  │  │
│           │  │ • legal_bert (nlpaueb/legal-bert-base-uncased)                 │  │  │
│           │  │ • bert (bert-base-uncased)                                      │  │  │
│           │  │ • roberta (roberta-base)                                        │  │  │
│           │  │ • deberta (microsoft/deberta-v3-base)                           │  │  │
│           │  │ • sentence_bert (sentence-transformers/all-mpnet-base-v2)       │  │  │
│           │  └────────────────────────────────────────────────────────────────┘  │  │
│           └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                                │
│                                    ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ INPUT: {text, label} samples                                                 │   │
│  │ PROCESS: Tokenization → Transformer Forward Pass → [CLS] / Mean Pooling      │   │
│  │ OUTPUT: (N, embedding_dim) embeddings                                        │   │
│  │ CACHE: storage/embeddings/{model}_{dataset}_{split}.pkl                       │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Spike Encoding                                                              │
│ ───────────────────────────────────────────────────────────────────────────────────  │
│                                                                                     │
│  CLI Args: --encodings poisson_rate latency temporal population binary_threshold   │
│           ┌─────────────────────────────────────────────────────────────────────┐  │
│           │                    ENCODERS Registry                                 │  │
│           │  ┌────────────────────────────────────────────────────────────────┐  │  │
│           │  │ 1. PoissonRateEncoder    - Rate ∝ activation value              │  │  │
│           │  │    • Firing rate proportional to embedding magnitude            │  │  │
│           │  │    • Decode: mean firing rate                                   │  │  │
│           │  │                                                                  │  │  │
│           │  │ 2. LatencyEncoder        - Time-to-first-spike coding            │  │  │
│           │  │    • High activation → early spike time                         │  │  │
│           │  │    • Decode: weighted sum with exponential decay                │  │  │
│           │  │                                                                  │  │  │
│           │  │ 3. TemporalEncoder       - Temporal contrast coding              │  │  │
│           │  │    • Spike at quantized time bin based on activation level       │  │  │
│           │  │    • Decode: weighted sum over time bins                         │  │  │
│           │  │                                                                  │  │  │
│           │  │ 4. PopulationEncoder    - Gaussian population coding              │  │  │
│           │  │    • Receptive field activation across neuron population         │  │  │
│           │  │    • Decode: weighted sum across population                      │  │  │
│           │  │                                                                  │  │  │
│           │  │ 5. BinaryThresholdEncoder - Binary threshold coding             │  │  │
│           │  │    • Active/inactive based on percentile threshold               │  │  │
│           │  │    • Decode: mean firing rate                                   │  │  │
│           │  └────────────────────────────────────────────────────────────────┘  │  │
│           └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                                │
│                                    ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ INPUT: (N, embedding_dim) embeddings                                          │   │
│  │ PROCESS: Min-max normalize → Apply encoding method → Binary spike trains     │   │
│  │ OUTPUT: (N, time_steps, embedding_dim) spike trains                          │   │
│  │ CONFIG: time_steps = 50 (default)                                            │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│   STAGE 4:            │ │   STAGE 5:            │ │   STAGE 6:            │
│   Transformer         │ │   SNN Training        │ │   Semantic            │
│   Baseline            │ │                       │ │   Preservation        │
└───────────────────────┘ └───────────────────────┘ └───────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Transformer Baseline Evaluation                                            │
│ ───────────────────────────────────────────────────────────────────────────────────  │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ TransformerBaseline.train_linear_probe()                                      │   │
│  │                                                                               │   │
│  │ PROCESS:                                                                      │   │
│  │ 1. Extract frozen embeddings from transformer                                 │   │
│  │ 2. Train Logistic Regression classifier on embeddings                       │   │
│  │ 3. Evaluate on validation set                                                 │   │
│  │                                                                               │   │
│  │ METRICS: accuracy, f1_macro, f1_micro, precision, recall                     │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: SNN Training                                                                │
│ ───────────────────────────────────────────────────────────────────────────────────  │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ SNNClassifier (per encoding method)                                           │   │
│  │                                                                               │   │
│  │ ARCHITECTURE (snntorch-based):                                                │   │
│  │ • Input: (time_steps, batch, features) spike trains                         │   │
│  │ • Hidden Layers: 2× FC(256) + Leaky LIF neurons                              │   │
│  │ • Output: FC(num_classes) + Leaky LIF                                       │   │
│  │ • Neuron: beta=0.9, threshold=1.0, reset=subtract                            │   │
│  │                                                                               │   │
│  │ TRAINING:                                                                     │   │
│  │ • Loss: CrossEntropyLoss on rate-coded output (sum over time)                │   │
│  │ • Optimizer: Adam (lr=1e-3)                                                  │   │
│  │ • Scheduler: Cosine annealing                                                │   │
│  │ • Epochs: 10 (default)                                                      │   │
│  │                                                                               │   │
│  │ METRICS: accuracy, f1_macro, f1_micro                                        │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: Semantic Preservation Analysis (RQ2 / H2)                                 │
│ ───────────────────────────────────────────────────────────────────────────────────  │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ SemanticPreservation.compare_encodings()                                      │   │
│  │                                                                               │   │
│  │ PROCESS:                                                                      │   │
│  │ 1. Decode spike trains back to embedding space (using encoder.decode())     │   │
│  │ 2. Compare decoded vs original transformer embeddings                         │   │
│  │                                                                               │   │
│  │ METRICS:                                                                      │   │
│  │ • Mean cosine similarity (paired samples)                                    │   │
│  │ • Spearman rank correlation of pairwise distances                            │   │
│  │ • Kendall-τ rank correlation                                                 │   │
│  │ • Top-k nearest neighbor overlap (k=5,10,20)                                 │   │
│  │ • Normalized MSE                                                              │   │
│  │                                                                               │   │
│  │ HYPOTHESIS H2: Time-based encodings (latency, temporal) preserve semantics   │   │
│  │              better than rate coding (poisson, binary)                       │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: Energy Analysis (RQ4 / H3)                                                 │
│ ───────────────────────────────────────────────────────────────────────────────────  │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ EnergyAnalyzer.analyze_all_encodings()                                       │   │
│  │                                                                               │   │
│  │ ENERGY MODEL (Horowitz 2014):                                                │   │
│  │ • Transformer MAC energy: 4.6 pJ per operation (GPU)                         │   │
│  │ • SNN SOP energy: 0.9 pJ per synaptic operation (neuromorphic)             │   │
│  │ • DRAM access: 100 pJ per access                                             │   │
│  │                                                                               │   │
│  │ TRANSFORMER ENERGY:                                                           │   │
│  │ • Compute: MAC operations × 4.6 pJ                                           │   │
│  │ • Memory: parameters × dtype_bytes × 100 pJ / 8                             │   │
│  │                                                                               │   │
│  │ SNN ENERGY:                                                                   │   │
│  │ • Compute: SOPs × 0.9 pJ                                                    │   │
│  │ • Memory: params × 4 × 100 pJ / 8 × active_fraction                        │   │
│  │ • Active fraction = 1 - sparsity (event-driven benefit)                      │   │
│  │                                                                               │   │
│  │ OUTPUT: Energy ratio (transformer/SNN), savings %, memory penalty           │   │
│  │                                                                               │   │
│  │ HYPOTHESIS H3: Energy improvements exist but shrink when memory access       │   │
│  │              is included in the analysis                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 8: Domain Shift Evaluation (RQ3 / H4)                                          │
│ ───────────────────────────────────────────────────────────────────────────────────  │
│                                                                                     │
│  CLI Args: --target-dataset ecthr_a                                                  │
│           ┌─────────────────────────────────────────────────────────────────────┐  │
│           │                    DomainShiftEvaluator                              │  │
│           └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                                │
│                                    ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ ZERO-SHOT TRANSFER PROTOCOL:                                                  │   │
│  │                                                                               │   │
│  │ 1. Load target dataset (e.g., ecthr_a)                                       │   │
│  │ 2. Extract target embeddings with same transformer                           │   │
│  │ 3. For each encoding:                                                        │   │
│  │    a. Encode target embeddings → spike trains                                │   │
│  │    b. Decode spike trains back to embedding space                           │   │
│  │ 4. Train linear probe on SOURCE dataset representations                      │   │
│  │ 5. Evaluate on TARGET dataset representations (no retraining)                 │   │
│  │                                                                               │   │
│  │ MODELS COMPARED:                                                              │   │
│  │ • transformer_legal_bert (raw embeddings)                                    │   │
│  │ • snn_poisson_rate (decoded)                                                 │   │
│  │ • snn_latency (decoded)                                                       │   │
│  │ • snn_temporal (decoded)                                                      │   │
│  │ • snn_population (decoded)                                                    │   │
│  │ • snn_binary_threshold (decoded)                                             │   │
│  │                                                                               │   │
│  │ METRICS:                                                                      │   │
│  │ • Source accuracy, Target accuracy                                           │   
│  │ • Accuracy drop, F1 drop                                                     │   │
│  │ • Relative drop %                                                             │   │
│  │ • H-score (harmonic mean of source and target)                               │   │
│  │                                                                               │   │
│  │ ADDITIONAL ANALYSIS:                                                          │   │
│  │ • Embedding shift: MMD, centroid distance, variance ratio                   │   │
│  │                                                                               │   │
│  │ HYPOTHESIS H4: Domain shift affects all models differently                    │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 9: Visualization & Reporting                                                   │
│ ───────────────────────────────────────────────────────────────────────────────────  │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ ResearchPlotter (matplotlib + seaborn)                                        │   │
│  │                                                                               │   │
│  │ FIGURES GENERATED:                                                            │   │
│  │ • Classification comparison bar chart                                        │   │
│  │ • Spike raster plots (per encoding)                                          │   │
│  │ • Firing rate violin plots                                                   │   │
│  │ • Semantic preservation radar/bar charts                                     │   │
│  │ • Energy comparison bar charts                                                │   │
│  │ • Energy-sparsity tradeoff scatter plot                                      │   │
│  │ • Domain shift heatmap                                                        │   │
│  │ • Summary dashboard (4-panel)                                                │   │
│  │                                                                               │   │
│  │ OUTPUT: storage/results/figures/*.pdf                                        │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                                │
│                                    ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ ReportGenerator                                                               │   │
│  │                                                                               │   │
│  │ FORMATS: HTML, Markdown, LaTeX                                                │   │
│  │                                                                               │   │
│  │ CONTENTS:                                                                     │   │
│  │ • Experiment configuration summary                                            │   │
│  │ • Classification results table                                               │   │
│  │ • Semantic preservation analysis                                             │   │
│  │ • Energy analysis with comparisons                                            │   │
│  │ • Domain shift evaluation results                                            │   │
│  │ • All figures embedded                                                        │   │
│  │                                                                               │   │
│  │ OUTPUT: storage/results/reports/report_*.html                                 │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
research/
├── main.py                          # CLI entry point (Click)
├── config.yaml                      # Configuration file
├── requirements.txt                 # Python dependencies
└── src/
    ├── __init__.py
    ├── datasets/                    # Dataset management
    │   ├── __init__.py
    │   ├── manager.py              # DatasetManager class
    │   ├── preprocessing.py        # Text preprocessing
    │   └── statistics.py           # DatasetStatistics class
    ├── encoding/                    # Spike encoding methods
    │   ├── __init__.py             # ENCODERS registry
    │   ├── base.py                 # BaseSpikeEncoder abstract class
    │   ├── poisson.py              # PoissonRateEncoder
    │   ├── latency.py              # LatencyEncoder
    │   ├── temporal.py             # TemporalEncoder
    │   ├── population.py           # PopulationEncoder
    │   └── binary.py               # BinaryThresholdEncoder
    ├── models/                      # Model implementations
    │   ├── __init__.py
    │   ├── transformer_baseline.py # TransformerBaseline class
    │   └── snn_model.py            # SNNClassifier class
    ├── evaluation/                  # Evaluation metrics
    │   ├── __init__.py
    │   ├── metrics.py              # ClassificationMetrics
    │   ├── semantic.py             # SemanticPreservation
    │   ├── energy.py               # EnergyAnalyzer
    │   └── domain_shift.py         # DomainShiftEvaluator
    ├── experiments/                 # Pipeline orchestration
    │   ├── __init__.py
    │   └── pipeline.py             # ExperimentPipeline class
    ├── visualization/               # Visualization
    │   ├── __init__.py
    │   └── plots.py                # ResearchPlotter class
    └── reporting/                   # Report generation
        ├── __init__.py
        └── report.py               # ReportGenerator class
```

## Research Questions & Hypotheses Mapping

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ RESEARCH QUESTIONS                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

RQ1: Which spike encoding performs best for legal text classification?
     └─ ADDRESSED IN: Stage 4 (Transformer) + Stage 5 (SNN) + Stage 9 (Visualization)
     └─ METRICS: accuracy, f1_macro, f1_micro
     └─ COMPARISON: transformer vs snn_{encoding} for each encoding method

RQ2: Does spike encoding preserve semantic similarity vs transformer embeddings?
     └─ ADDRESSED IN: Stage 6 (Semantic Preservation)
     └─ METRICS: cosine similarity, spearman rho, kendall tau, top-k NN overlap
     └─ HYPOTHESIS H2: Time-based encodings > rate coding for semantic preservation

RQ3: How robust are spike-based classifiers under domain shift?
     └─ ADDRESSED IN: Stage 8 (Domain Shift)
     └─ METRICS: accuracy drop, H-score, embedding shift (MMD)
     └─ PROTOCOL: Train on source, evaluate on target (zero-shot transfer)

RQ4: Does spike encoding provide measurable energy savings vs transformer baselines?
     └─ ADDRESSED IN: Stage 7 (Energy Analysis)
     └─ METRICS: energy ratio, savings %, memory penalty
     └─ HYPOTHESIS H3: Savings shrink when memory access is included
```

## Data Flow Diagram

```
Raw Legal Text
     │
     ▼
┌──────────────┐
│ Dataset Load │────▶ {train: [{text, label}], val: [...], test: [...]}
└──────────────┘
     │
     ▼
┌──────────────┐
│ Tokenization │────▶ input_ids, attention_mask
└──────────────┘
     │
     ▼
┌──────────────┐
│ Transformer  │────▶ (N, 768) embeddings [CLS] token
│  Forward     │
└──────────────┘
     │
     ├──────────────────────────────────────┐
     │                                      │
     ▼                                      ▼
┌──────────────┐                    ┌──────────────┐
│ Linear Probe │                    │ Spike Encode │
│ (Baseline)   │                    │ (5 methods)  │
└──────────────┘                    └──────────────┘
     │                                      │
     ▼                                      ▼
Classification Metrics              (N, 50, 768) Spike Trains
     │                                      │
     │                                      ▼
     │                              ┌──────────────┐
     │                              │ SNN Training │
     │                              │ (per enc)    │
     │                              └──────────────┘
     │                                      │
     │                                      ▼
     │                              Classification Metrics
     │                                      │
     │                                      ├──────────────────┐
     │                                      │                  │
     │                                      ▼                  ▼
     │                              ┌──────────────┐  ┌──────────────┐
     │                              │   Semantic   │  │    Energy    │
     │                              │  Preservation │  │   Analysis   │
     │                              └──────────────┘  └──────────────┘
     │                                      │                  │
     └──────────────────────────────────────┼──────────────────┘
                                            │
                                            ▼
                                      ┌──────────────┐
                                      │  Domain Shift│
                                      │  (optional)  │
                                      └──────────────┘
                                            │
                                            ▼
                                      ┌──────────────┐
                                      │ Visualization│
                                      │  + Reporting │
                                      └──────────────┘
```

## Storage Structure

```
storage/
├── datasets/
│   ├── raw/                    # Original downloaded data
│   ├── processed/              # Preprocessed splits
│   └── cache/                  # Pickled dataset cache
│       ├── case_hold.pkl
│       ├── ecthr_a.pkl
│       └── custom_*.pkl
├── embeddings/                 # Cached transformer embeddings
│   ├── legal_bert_case_hold_train.pkl
│   ├── legal_bert_case_hold_validation.pkl
│   └── legal_bert_case_hold_test.pkl
├── checkpoints/                # Model checkpoints
└── results/
    ├── figures/                # Generated visualizations (PDF)
    │   ├── classification_comparison.pdf
    │   ├── encoding_comparison_rasters.pdf
    │   ├── semantic_preservation.pdf
    │   ├── energy_comparison.pdf
    │   └── summary_dashboard.pdf
    ├── reports/                # Generated reports
    │   ├── report_case_hold_legal_bert.html
    │   └── report_case_hold_legal_bert.md
    ├── results_*.json          # Full raw results JSON
    └── experiment.log          # Experiment log file
```

## Key Configuration Parameters

```yaml
# Dataset Configuration
datasets:
  max_train_samples: 500
  max_val_samples: 100
  max_test_samples: 200

# Encoding Configuration
encoding:
  time_steps: 50
  methods:
    poisson_rate: {enabled: true, max_rate: 100}
    latency: {enabled: true, tau: 5.0}
    temporal: {enabled: true, n_levels: 10}
    population: {enabled: true, n_neurons: 10}
    binary_threshold: {enabled: true, percentile: 50.0}

# SNN Configuration
snn:
  architecture:
    hidden_size: 256
    num_hidden_layers: 2
    dropout: 0.3
  neuron:
    type: "leaky"
    beta: 0.9
    threshold: 1.0
  training:
    learning_rate: 1.0e-3
    batch_size: 32
    num_epochs: 10

# Energy Configuration
evaluation:
  energy:
    mac_energy_pj: 4.6        # GPU MAC energy
    sop_energy_pj: 0.9        # Neuromorphic SOP energy
    memory_access_energy_pj: 100.0  # DRAM access energy
```

## CLI Commands

```bash
# Run full pipeline
python main.py run --dataset case_hold --encoder legal_bert

# Quick test run
python main.py run --quick

# Specific encodings only
python main.py run --encodings poisson_rate latency

# Skip stages
python main.py run --skip snn --skip domain_shift

# Domain shift evaluation
python main.py run --target-dataset ecthr_a

# Dataset management
python main.py dataset list
python main.py dataset info case_hold
python main.py dataset download case_hold

# Spike encoding demo
python main.py encode demo
python main.py encode compare --dataset case_hold

# Regenerate report
python main.py report --format latex
```

## Error Handling & Current Issue

The current error occurs in Stage 1 (Dataset Loading) due to a pandas import issue:

```
ImportError: DLL load failed while importing conversion: 
An Application Control policy has blocked this file.
```

This is a system-level security policy issue preventing pandas from loading its C extensions.
The error is not in the codebase logic but in the Python environment configuration.

**Solution**: The pandas library needs to be reinstalled or the system's Application Control
policy needs to be adjusted to allow pandas DLL files to load.
