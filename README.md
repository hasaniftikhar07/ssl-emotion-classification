# Semi-Supervised Emotion Classification

This repository contains the implementation and experiments for a research project on semi-supervised emotion classification using classical SSL methods and transformer baselines.

## 📄 Associated Research Paper

**Title:** Semi-Supervised Emotion Classification under Limited Labeled Data:  
A Comparative Study of Classical SSL Methods and Transformer Models.

The project evaluates multiple semi-supervised learning algorithms under a strict inductive evaluation protocol and compares their performance with a transformer-based baseline.

Manuscript under journal submission.
Preprint version available in this repository.

If you use this code, please cite the associated paper.

The repository includes the full experimental pipeline, reproducibility scripts, and the manuscript associated with the project.

## Repository Structure

data/  
Dataset documentation and preprocessing.

ssl_pipeline.py  
Core semi-supervised learning pipeline.

run_experiment.py  
Runs experiments and produces evaluation metrics.

requirements.txt  
Python dependencies required to run the experiments.

ssl_emotion_classification.pdf  
Research manuscript associated with this project.

Models implemented:

- Self-Training SVM
- Co-Training
- S3VM
- Label Propagation
- Label Spreading
- DistilRoBERTa Transformer

Dataset size: 23,485 reviews

Author: Hasan Iftikhar

## Research Paper

Manuscript currently under journal submission.

A preprint version of the paper is included in this repository.

## Citation

If you use this code or dataset in your research, please cite:

Iftikhar, H. (2026) *Semi-Supervised Emotion Classification under Limited Labeled Data: A Comparative Study of Classical SSL Methods and Transformer Models.* Manuscript under journal submission.

### BibTeX

@article{iftikhar2026ssl,
title={Semi-Supervised Emotion Classification under Limited Labeled Data: A Comparative Study of Classical SSL Methods and Transformer Models},
author={Iftikhar, Hasan},
year={2026},
note={Manuscript under journal submission}
}

## Key Results

Experiments were conducted on 23,485 eCommerce reviews across 7 emotion classes.

| Model | Accuracy | Macro F1 |
|------|------|------|
| Self-Training SVM | 0.68 | 0.59 |
| Co-Training (LR+RF) | 0.62 | 0.13 |
| Co-Training (GB+SVM) | 0.71 | 0.53 |
| **S3VM** | **0.97** | **0.88** |
| Label Propagation (RBF) | 0.60 | 0.11 |
| Label Spreading (RBF) | 0.60 | 0.11 |
| Label Propagation (KNN) | 0.01 | 0.01 |
| Label Spreading (KNN) | 0.01 | 0.01 |
| DistilRoBERTa | 0.89 | 0.86 |

## Evaluation Protocol

Graph-based semi-supervised models were evaluated using a strict inductive protocol:

• 80/20 stratified split of labeled data  
• test set excluded from graph construction  
• TF-IDF fitted only on training + unlabeled pool  
• predictions made using `.predict()` on unseen data

This prevents transductive leakage where test samples influence the propagation graph.


## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run experiment:

```bash
python run_experiment.py
```






