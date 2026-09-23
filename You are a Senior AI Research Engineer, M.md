You are a Senior AI Research Engineer, Machine Learning Researcher, Python Software Engineer, and Scientific Computing Expert.

Your task is to build a COMPLETE experimental research framework for my research project.

This framework is NOT intended to be a commercial application.

It is NOT an enterprise system.

It is NOT a SaaS product.

It is NOT a production deployment platform.

Instead, it is a research framework whose only purpose is to conduct reproducible experiments, compare models, generate evaluation metrics, create publication-quality visualizations, and produce the experimental results required for my research paper.

====================================================================
RESEARCH TITLE
====================================================================

A Semantic- and Energy-Aware Study of Spike Encoding for Legal Text Classification Under Domain Shift

====================================================================
PRIMARY OBJECTIVE
====================================================================

Develop a complete experimental pipeline that compares Transformer-based Legal NLP models with Spike Encoding + Spiking Neural Networks.

The framework must allow me to conduct all experiments required to answer my research questions and validate my hypotheses.

The software itself is only a research tool.

The outputs of the framework will become the experimental results, tables, graphs, and conclusions of my research paper.

====================================================================
RESEARCH QUESTIONS
====================================================================

The framework must enable experiments that answer:

• Which spike encoding performs best for legal text classification?

• Does spike encoding preserve semantic similarity compared to the original transformer embeddings?

• How robust are spike-based legal classifiers under domain shift?

• Does spike encoding provide measurable energy savings compared to transformer baselines?

====================================================================
HYPOTHESES
====================================================================

The framework should allow validation of:

H1
Spike encoding achieves competitive classification performance compared to transformer baselines.

H2
Time-based spike encodings preserve legal semantics better than simple rate coding.

H3
Energy improvements exist but become smaller when memory access is included.

H4
Domain shift affects all models differently.

====================================================================
IMPLEMENTATION REQUIREMENTS
====================================================================

Everything must be written entirely in Python.

No external infrastructure.

Do NOT use:

PostgreSQL

MySQL

MongoDB

Redis

Celery

RabbitMQ

Kafka

Docker

Docker Compose

Kubernetes

MLflow

Cloud databases

Cloud storage

Cloud experiment tracking

The framework must run after only

pip install -r requirements.txt

python main.py

====================================================================
HUGGING FACE INTEGRATION
====================================================================

Use the Hugging Face ecosystem throughout the project.

Libraries

datasets

transformers

sentence-transformers

huggingface_hub

Automatically download and locally cache all datasets and models.

Support offline execution after the first download.

Support:

LexGLUE

CaseHOLD

ECtHR-A

ECtHR-B

EURLEX

LEDGAR

SCOTUS

UNFAIR-ToS

LEXTREME

Also allow custom datasets:

CSV

JSON

JSONL

Parquet

Excel

Support Hugging Face transformer models including:

LegalBERT

BERT

RoBERTa

DeBERTa-v3

Sentence-BERT

Automatically cache all downloaded models locally.
====================================================================
DATASET MANAGEMENT
====================================================================

The framework must use the Hugging Face `datasets` library for downloading, loading, preprocessing, and caching legal NLP datasets.

Import using

from datasets import load_dataset

The Dataset Manager must automatically download datasets the first time they are used and cache them locally for future experiments.

Downloaded datasets should never be downloaded again unless the user explicitly refreshes the cache.

Support both online and offline execution after the first download.

====================================================================
SUPPORTED DATASETS
====================================================================

Initially implement support for the following legal datasets.

1. CaseHOLD

```python
from datasets import load_dataset

dataset = load_dataset(
    "coastalcph/lex_glue",
    "case_hold"
)
```

Use for

• Long legal document classification
• Transformer baseline
• Spike encoding experiments
• Domain shift evaluation

------------------------------------------------------------

2. ECtHR-A

```python
from datasets import load_dataset

dataset = load_dataset(
    "coastalcph/lex_glue",
    "ecthr_a"
)
```

Use for

• Multi-label legal classification
• Cross-domain evaluation
• Semantic preservation experiments

------------------------------------------------------------

3. ECtHR-B

```python
from datasets import load_dataset

dataset = load_dataset(
    "coastalcph/lex_glue",
    "ecthr_b"
)
```

Use for

• Cross-domain robustness
• Generalization experiments
• Domain shift testing

====================================================================
FUTURE DATASET SUPPORT
====================================================================

The Dataset Manager should be modular so that new datasets can be added with minimal code changes.

Prepare support for

• LexGLUE
• LEXTREME
• EURLEX
• LEDGAR
• SCOTUS
• UNFAIR-ToS

The framework should also allow users to add any Hugging Face dataset by entering the dataset name and configuration.

====================================================================
CUSTOM DATASETS
====================================================================

Support uploading local datasets in the following formats

• CSV
• JSON
• JSONL
• Excel (.xlsx)
• Parquet

Automatically detect

• Text column
• Label column
• Multi-label datasets
• Class names

====================================================================
DATASET MANAGER FEATURES
====================================================================

Implement a complete Dataset Manager capable of

• Automatic downloading
• Automatic local caching
• Dataset version tracking
• Dataset preview
• Train/Validation/Test split viewer
• Dataset statistics
• Number of samples
• Number of classes
• Label distribution
• Class imbalance statistics
• Average document length
• Median document length
• Maximum document length
• Token length statistics
• Vocabulary statistics
• Search and filtering
• Dataset refresh
• Delete cached datasets
• Export dataset statistics

====================================================================
PREPROCESSING
====================================================================

The Dataset Manager should provide configurable preprocessing including

• Text cleaning
• Unicode normalization
• Lowercasing (optional)
• Stopword removal (optional)
• Tokenization
• Padding
• Truncation
• Sliding window chunking for long legal documents
• Batch processing

====================================================================
CACHE
====================================================================

Store datasets locally in

storage/

    datasets/

        raw/

        processed/

        cache/

The framework should automatically reuse cached datasets whenever possible to ensure experiments are reproducible and avoid unnecessary downloads.
====================================================================
EXPERIMENTAL PIPELINE
====================================================================

The framework should implement the following workflow:

Dataset Selection

↓

Dataset Download

↓

Preprocessing

↓

Transformer Tokenization

↓

Embedding Generation

↓

Embedding Cache

↓

Spike Encoding

↓

Spike Train Generation

↓

Transformer Baseline Training

↓

SNN Training

↓

Evaluation

↓

Semantic Preservation Analysis

↓

Energy Analysis

↓

Domain Shift Evaluation

↓

Visualization

↓

Automatic Report Generation

====================================================================
SPIKE ENCODING
====================================================================

Implement multiple encoding methods including

Poisson Rate Coding

Latency Coding

Temporal Coding

Population Coding

Binary Threshold Encoding

Each encoding should be independently configurable and comparable.

====================================================================
SNN IMPLEMENTATION
====================================================================

Support

snnTorch

Norse

SpikingJelly

Neuron Models

IF

LIF

Adaptive LIF

Support configurable

timesteps

thresholds

surrogate gradients

membrane decay

====================================================================
TRANSFORMER BASELINES
====================================================================

Implement baseline experiments using

LegalBERT

BERT

RoBERTa

DeBERTa

Train using the same dataset splits as the SNN experiments.

====================================================================
EVALUATION
====================================================================

Automatically compute

Accuracy

Precision

Recall

Macro F1

Micro F1

Weighted F1

ROC AUC

PR Curve

Confusion Matrix

====================================================================
SEMANTIC PRESERVATION
====================================================================

Compare transformer embeddings with spike representations using

Cosine Similarity

Euclidean Distance

Linear CKA

RBF CKA

Centered Cosine Similarity

Representational Similarity Analysis

====================================================================
ENERGY ANALYSIS
====================================================================

Estimate energy using

Spike Count

Memory Access

Multiply-Accumulate Operations

Accumulate Operations

Inference Time

Training Time

CPU Usage

GPU Usage

RAM Usage

The implementation should follow the energy-aware methodology proposed in my research rather than relying only on spike counts.

====================================================================
DOMAIN SHIFT
====================================================================

Support cross-dataset evaluation.

Examples

Train on CaseHOLD

↓

Evaluate on ECtHR-A

Train on ECtHR-A

↓

Evaluate on ECtHR-B

Generate transfer matrices showing robustness.

====================================================================
EXPERIMENT STORAGE
====================================================================

Store every experiment locally.

Example

experiments/

experiment_001/

config.json

metrics.json

history.csv

checkpoint.pt

predictions.csv

plots/

logs.txt

report.pdf

latex_tables/

No SQL database should be used.

====================================================================
VISUALIZATION
====================================================================

Generate publication-quality figures.

Support

Loss curves

Accuracy curves

ROC curves

PR curves

Confusion matrices

Spike rasters

Spike histograms

Embedding visualizations

t-SNE

UMAP

Semantic preservation plots

Energy comparison graphs

Domain shift heatmaps

====================================================================
REPORT GENERATION
====================================================================

Automatically generate

CSV

Excel

JSON

Markdown

PDF

LaTeX

The generated figures and tables should be directly usable in my research paper.

====================================================================
CODING REQUIREMENTS
====================================================================

Write real code only.

No placeholders.

No TODOs.

No fake metrics.

Every experiment should be executable.

Every metric must come from actual computation.

Use modular architecture.

Use type hints.

Use docstrings.

Use structured logging.

Write clean, reusable research code.

====================================================================
FINAL GOAL
====================================================================

The completed framework should allow me to conduct every experiment described in my research proposal, generate reproducible results, compare spike encoding methods with transformer baselines, evaluate semantic preservation, energy efficiency, and domain shift robustness, and automatically produce publication-ready graphs, tables, and reports for inclusion in my thesis.