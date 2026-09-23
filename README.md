# Spike-Legal-NLP

## A Semantic- and Energy-Aware Study of Spike Encoding for Legal Text Classification Under Domain Shift

Spike-Legal-NLP is a research framework for studying spike encoding strategies for legal text classification using Spiking Neural Networks (SNNs). The framework compares transformer-based baselines with five spike encoding strategies across classification performance, semantic preservation, theoretical energy consumption, and domain-shift robustness.

## Research Questions

- **RQ1:** How do different spike encoding schemes perform for legal text classification?
- **RQ2:** How well does spike encoding preserve the semantic structure of transformer embeddings?
- **RQ3:** How robust are spike-based classifiers under domain shift?
- **RQ4:** What theoretical energy savings can be obtained compared with transformer-based computation?

## Spike Encoding Strategies

The framework evaluates five encoding methods:

1. **Poisson Rate Coding** — represents embedding values as firing rates and generates stochastic spike trains.
2. **Latency Coding** — represents embedding values through spike timing.
3. **Temporal Coding** — quantizes values into discrete levels represented at different time steps.
4. **Population Coding** — represents scalar values using multiple neurons with receptive fields.
5. **Adaptive Binary-Threshold Coding** — generates spikes when values exceed a percentile-based adaptive threshold.

## System Pipeline

Legal Text → LegalBERT / BERT → Dense Embedding → Spike Encoding → LIF SNN Classifier → Multi-Axis Evaluation

The spike encoding stage contains:

- Poisson Rate
- Latency
- Temporal
- Population
- Binary Threshold

The final evaluation covers:

- Classification Performance
- Semantic Preservation
- Theoretical Energy Consumption
- Domain-Shift Robustness

## Methodology

The Spike-Legal-NLP framework follows a four-stage pipeline:

1. Dense sentence/document embeddings are extracted from a pretrained transformer such as LegalBERT or BERT.
2. The embeddings are converted into spike trains using one of the five encoding strategies.
3. A lightweight SNN classifier based on Leaky Integrate-and-Fire (LIF) neurons is trained on the spike trains.
4. The resulting models are evaluated across classification performance, semantic preservation, theoretical energy consumption, and domain-shift robustness.

## Datasets

The experiments use legal NLP datasets from the **LexGLUE** benchmark.

### CaseHOLD

CaseHOLD is used for the main legal text classification experiment.

### ECtHR-A and ECtHR-B

ECtHR-A and ECtHR-B are used for the revised domain-shift evaluation.

The final domain-shift protocol trains on **ECtHR-A** and evaluates zero-shot transfer on **ECtHR-B** because these datasets provide the label-compatible pair supported by the framework.

## Model Architecture

The SNN classifier is a feed-forward network using **Leaky Integrate-and-Fire (LIF) neurons** with a fixed decay constant, firing threshold, and subtractive reset mechanism.

Training uses:

- Adam optimizer
- Cosine learning-rate schedule
- Surrogate-gradient-based training
- LIF neuron dynamics

The same downstream classifier architecture is used across the encoding strategies.

## Evaluation Metrics

### Classification Performance

- Accuracy
- Macro F1
- Micro F1

### Semantic Preservation

- Cosine similarity
- Spearman rank correlation
- Kendall rank correlation
- Top-10 nearest-neighbour overlap

### Domain-Shift Robustness

- Source accuracy
- Target accuracy
- Accuracy drop
- F1 drop
- H-score

### Energy Evaluation

The framework estimates energy consumption using:

- Multiply-Accumulate (MAC) operations
- Synaptic Operations (SOP)
- DRAM/memory-access costs

The energy analysis is theoretical and is not a measurement performed on physical neuromorphic hardware.

## Experimental Configuration

| Parameter | Value |
|---|---:|
| SNN Hidden Size | 512 |
| SNN Epochs | Up to 40 |
| SNN Beta (LIF Decay) | 0.99 |
| SNN Batch Size | 32 |
| Maximum Sequence Length | 512 |
| Source Dataset | CaseHOLD |
| Domain-Shift Source | ECtHR-A |
| Domain-Shift Target | ECtHR-B |

## CaseHOLD Classification Results

| Model / Encoding | Accuracy | F1-Macro | F1-Micro |
|---|---:|---:|---:|
| Transformer LegalBERT | 0.1860 | 0.1844 | 0.1860 |
| Poisson Rate | 0.2400 | 0.1991 | 0.2400 |
| Latency | 0.2100 | 0.1404 | 0.2100 |
| Temporal | 0.1920 | 0.1378 | 0.1920 |
| Population | 0.2040 | 0.1441 | 0.2040 |
| Binary Threshold | 0.1720 | 0.1646 | 0.1720 |

## Domain-Shift Results

The revised domain-shift experiment uses **ECtHR-A → ECtHR-B**.

| Model | Avg. Accuracy Drop | Avg. Target Accuracy | Avg. H-Score |
|---|---:|---:|---:|
| Transformer | -0.1722 | 0.5833 | 0.4823 |
| Poisson Rate | -0.1767 | 0.5767 | 0.4724 |
| Latency | -0.1722 | 0.5833 | 0.4823 |
| Temporal | -0.1833 | 0.5833 | 0.4746 |
| Population | -0.1833 | 0.5833 | 0.4746 |
| Binary Threshold | -0.1789 | 0.5900 | 0.4846 |

## Experimental Corrections

Several issues were identified during the research process and addressed in the corrected experiments:

- A classification readout issue affected the SNN predictions.
- The original per-sample min-max normalization was affected by outlier embedding dimensions.
- Global, per-feature normalization was introduced for the corrected experiments.
- An out-of-memory issue occurred during population-encoding training and was addressed by reducing the SNN training batch size.
- The original CaseHOLD → ECtHR domain-shift protocol contained a label-space mismatch.
- The domain-shift protocol was revised to the label-compatible ECtHR-A → ECtHR-B evaluation.

## Project Outputs

The framework generates research reports and visualizations including:

- Spike encoding comparison rasters
- Firing-rate plots
- Embedding scatter plots
- Semantic-preservation analysis
- Domain-shift evaluation
- Hypothesis evaluation
- Experimental result reports

## Limitations

- The dataset size may limit statistical power and generalizability.
- Energy estimates are theoretical calculations and are not measurements from physical neuromorphic hardware.
- Poisson-based encoding is stochastic, so repeated runs can produce different efficiency values.
- ECtHR-A and ECtHR-B share source documents and a label taxonomy, so the revised domain-shift experiment should not be interpreted as transfer between substantially different legal domains.
- The project contains discrepancies between some methodology-document and runtime-configuration hyperparameters.

## Framework

**Spike-Legal-NLP v1.0**

## Research Area

**Legal NLP · Spiking Neural Networks · Spike Encoding · Semantic Preservation · Energy-Aware Computing · Domain Shift**

## Author

**Payal Salve**  
Department of Computer Science  
Ramnarain Ruia Autonomous College

## Research Paper

**A Semantic- and Energy-Aware Study of Spike Encoding for Legal Text Classification Under Domain Shift**
