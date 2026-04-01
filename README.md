# AI Methods for Digital Fatigue and Digital Exhaustion Detection

Research code for a PhD dissertation in Computer Science focused on detecting digital fatigue and digital exhaustion in text-based online communication.  
The repository implements an end-to-end analytical workflow that combines neural text classification, communicative segmentation, and an integrated author-level profile.

## Research Context

The work addresses a practical and scientific problem: users increasingly operate in permanently connected digital environments, where cognitive and emotional overload may accumulate and evolve into stable fatigue and, later, digital exhaustion.

Unlike approaches based only on global activity indicators, this project models fatigue at the level of **local communicative segments** and then aggregates segment-level evidence into an interpretable **integrated exhaustion profile**.

## Research Contribution

This repository operationalizes three core methodological contributions:

1. **Communicative Segmentation Method**
   - User messages are grouped into semantically coherent segments using embeddings and density-based clustering.
   - Each segment is interpreted as a local source of cognitive load.

2. **Segment-Level Digital Fatigue Detection**
   - A deep learning binary classifier estimates fatigue probability for individual messages.
   - Segment-level fatigue indices are computed from message-level estimates and visualized for interpretation.

3. **Integrated Digital Exhaustion Profiling**
   - Local segment indices are aggregated into an author-level profile.
   - The profile distinguishes between normal state, situational fatigue, and systemic exhaustion using explicit thresholds.

## Methodology Overview

### Stage 1. Neural Classifier Training

- Input: labeled CSV dataset with text and binary target (`0/1`).
- Model family: Transformer-based sequence classification (`transformers`).
- Outputs: trained model, evaluation metrics, confusion matrix, per-sample predictions.

### Stage 2. Communicative Segmentation

- Input: CSV dataset with author identifier and text.
- Pipeline:
  - sentence embeddings (`sentence-transformers`);
  - manifold projection (`UMAP`);
  - density clustering (`HDBSCAN`);
  - segment naming via C-TF-IDF keywords.
- Outputs: segment map, inter-segment distance heatmap, segment summary table, representative texts.
- Optional control: post-clustering noise constraint (`max_noise_share`).

### Stage 3. Integrated Author Profile

- Uses outputs from stages 1 and 2.
- Computes:
  - local segment fatigue index `e_i`;
  - proportion of critical segments (`CoverageTheta`).
- Produces:
  - profile state and thresholds report;
  - gauge, bar plot, segment pies, and word clouds for interpretability.

## Repository Structure

- `notebooks/` - original research notebooks and experimental drafts.
- `src/digital_fatigue/` - production-like modular implementation:
  - `utils.py` - text normalization, stopwords, keyword filtering;
  - `segmentation.py` - segmentation, keyword extraction, segment diagnostics;
  - `training.py` - model training and evaluation;
  - `ui.py` - Gradio application and workflow orchestration.
- `scripts/run_app.py` - application entry point.
- `requirements.txt` - dependency specification.

## Data Requirements

### Training Dataset

- CSV file with:
  - text column;
  - binary label column (`0` = no explicit fatigue signs, `1` = fatigue signs).

### Segmentation Dataset

- CSV file with:
  - author identifier column;
  - text message column.

## Reproducibility and Execution

### Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Application

```bash
PYTHONPATH=src python scripts/run_app.py
```

Open the Gradio URL and execute the workflow sequentially:
**training -> segmentation -> integrated profile**.

## Interpretation Outputs

The interface is designed for research interpretation, not only prediction:

- named communicative segments and target keywords;
- nearest-segment geometry and centroid-distance maps;
- local fatigue index distributions across segments;
- integrated profile indicators (including threshold-based status).

## Limitations

- This repository is intended for research and dissertation evaluation workflows.
- Model quality is sensitive to annotation quality, class balance, and domain/language mismatch.
- Thresholds used in profile interpretation are configurable and should be validated for each dataset.