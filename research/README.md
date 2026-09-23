# Spike-Legal-NLP Research Framework

**A Semantic- and Energy-Aware Study of Spike Encoding for Legal Text Classification Under Domain Shift**

---

## Overview

This framework implements a complete experimental pipeline for comparing Transformer-based Legal NLP models with Spike Encoding + Spiking Neural Networks (SNNs). It produces all experimental results, tables, graphs, and conclusions needed for the research paper.

## Research Questions

| # | Question |
|---|----------|
| RQ1 | Which spike encoding performs best for legal text classification? |
| RQ2 | Does spike encoding preserve semantic similarity vs transformer embeddings? |
| RQ3 | How robust are spike-based classifiers under domain shift? |
| RQ4 | Does spike encoding provide measurable energy savings vs transformer baselines? |

## Hypotheses

- **H1**: Spike encoding achieves competitive classification performance vs transformer baselines
- **H2**: Time-based spike encodings preserve legal semantics better than simple rate coding
- **H3**: Energy improvements exist but become smaller when memory access is included
- **H4**: Domain shift affects all models differently

---

## Setup

```bash
cd research
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline
```bash
#RUN KR RHA THA BC BHOT TIME LAG RHA AHI
# Full pipeline on CaseHOLD with LegalBERT
python main.py run --dataset case_hold --encoder legal_bert

# Quick test run (reduced samples)
python main.py run --quick

# Only specific spike encodings NAHI RUN HUA BC
python main.py run --encodings poisson_rate latency temporal

# Skip slow stages
python main.py run --skip snn --skip domain_shift

# Cap samples per split (CPU-friendly medium run)
python main.py run --max-train-samples 2000 --max-val-samples 500 --max-test-samples 500

# File-only logging (no console output)
python main.py --quiet run
```

Progress is printed to the console and mirrored to `storage/results/experiment.log`.
Embedding extraction shows a per-batch progress bar and is cached per
(model, dataset, split, sample count) under `storage/embeddings/`.

On CPU, LegalBERT embeds roughly 8-10 samples/second at `max_length: 512`, so the
full 45k-sample CaseHOLD training split takes hours; use the sample caps above
unless running on a GPU. Spike encoding allocates a
`(samples, time_steps, features)` float32 tensor (times `n_neurons` for population
coding), and the run aborts with a clear error if one split exceeds
`encoding.max_memory_gb`.

### Dataset management
```bash
python main.py dataset list                    # List all datasets
python main.py dataset info case_hold          # Show statistics
python main.py dataset download case_hold      # Pre-download
python main.py dataset custom data.csv         # Load custom datasetNAHI RUN HUA BC
```

### Spike encoding demo
```bash
python main.py encode demo                     # Demo on sample text
python main.py encode compare --dataset case_hold  # Compare all encodings
```

### Regenerate report
```bash
python main.py report                          # From latest results
python main.py report --format latex           # As LaTeX
python main.py report --format markdown        # As Markdown
```

---

## Pipeline Stages

```
Dataset Load → Transformer Embeddings → Spike Encoding → 
SNN Training → Semantic Analysis → Energy Analysis →
Visualization → HTML Report
```

| Stage | Description | Can Skip |
|-------|-------------|----------|
| `dataset` | Download & cache legal NLP dataset | — |
| `embeddings` | Extract transformer embeddings | `--skip embeddings` |
| `encoding` | Generate spike trains (all 5 methods) | — |
| `transformer_eval` | Train linear probe on embeddings | `--skip transformer_eval` |
| `snn` | Train SNN classifiers on spike trains | `--skip snn` |
| `semantic` | Semantic preservation analysis | `--skip semantic` |
| `energy` | Energy comparison analysis | `--skip energy` |

---

## Supported Datasets

| Key | Dataset | Task | Source |
|-----|---------|------|--------|
| `case_hold` | CaseHOLD | Multi-class | LexGLUE |
| `ecthr_a` | ECtHR-A | Multi-label | LexGLUE |
| `ecthr_b` | ECtHR-B | Multi-label | LexGLUE |
| `eurlex` | EURLEX | Multi-label | LexGLUE |
| `ledgar` | LEDGAR | Multi-class | LexGLUE |
| `scotus` | SCOTUS | Multi-class | LexGLUE |
| `unfair_tos` | UNFAIR-ToS | Multi-label | LexGLUE |

Custom datasets: CSV, JSON, JSONL, Excel (.xlsx), Parquet

---

## Supported Models

| Key | Model |
|-----|-------|
| `legal_bert` | nlpaueb/legal-bert-base-uncased |
| `bert` | bert-base-uncased |
| `roberta` | roberta-base |
| `deberta` | microsoft/deberta-v3-base |
| `sentence_bert` | sentence-transformers/all-mpnet-base-v2 |

---

## Spike Encoding Methods

| Method | Description | Key Idea |
|--------|-------------|---------|
| `poisson_rate` | Poisson Rate Coding | Rate ∝ activation value |
| `latency` | Latency (Time-to-First-Spike) | High activation → early spike |
| `temporal` | Temporal Contrast | Spike at quantized time bin |
| `population` | Gaussian Population Coding | Receptive field activation |
| `binary_threshold` | Binary Threshold | Active/inactive based on percentile |

---

## Output Structure

```
storage/
├── datasets/
│   ├── raw/                  # Raw downloaded data
│   ├── processed/            # Preprocessed splits
│   └── cache/                # Pickled dataset cache
├── embeddings/               # Cached transformer embeddings
├── checkpoints/              # Model checkpoints
└── results/
    ├── figures/              # All generated figures (PDF)
    ├── reports/              # HTML/Markdown/LaTeX reports
    ├── results_*.json        # Full raw results JSON
    └── experiment.log        # Experiment log
```

---

## Configuration

All settings are in `config.yaml`. Key parameters:

```yaml
datasets:
  max_train_samples: 500      # Limit training samples
  max_val_samples: 100

encoding:
  time_steps: 50              # Spike train length

snn:
  architecture:
    hidden_size: 256
  training:
    num_epochs: 10

evaluation:
  energy:
    mac_energy_pj: 4.6        # pJ/MAC (GPU, Horowitz 2014)
    sop_energy_pj: 0.9        # pJ/SOP (neuromorphic)
```

---

## Energy Model

Energy estimates follow Horowitz (2014):
- **Transformer (MAC)**: 4.6 pJ per multiply-accumulate on GPU
- **SNN (SOP)**: 0.9 pJ per synaptic operation on neuromorphic hardware
- **DRAM access**: 100 pJ per access

The analysis tests H3: energy savings shrink when memory bandwidth is included.
