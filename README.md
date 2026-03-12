# Semi-Supervised Emotion Classification

This repository contains the implementation and experiments for a research project on semi-supervised emotion classification using classical SSL methods and transformer baselines, with strict inductive evaluation of graph-based semi-supervised learning methods.

## 📝 Associated Research Paper

**Title:** Semi-Supervised Emotion Classification under Limited Labeled Data:  
A Comparative Study of Classical SSL Methods and Transformer Models

This paper evaluates multiple semi-supervised learning algorithms under limited labeled data conditions and includes a strict inductive evaluation of graph-based SSL methods.

Manuscript under journal submission.  
A preprint version is included in this repository.

If you use this code, please cite the associated paper.

## Key Results

Experiments were conducted on 23,485 eCommerce reviews across 7 emotion classes.

| Model | Accuracy | Macro F1 |
|------|------:|------:|
| Self-Training SVM | 0.68 | 0.59 |
| Co-Training (LR+RF) | 0.62 | 0.13 |
| Co-Training (GB+SVM) | 0.71 | 0.53 |
| **S3VM** | **0.97** | **0.88** |
| Label Propagation (RBF, inductive) | 0.60 | 0.11 |
| Label Spreading (RBF, inductive) | 0.60 | 0.11 |
| Label Propagation (KNN, inductive) | 0.01 | 0.01 |
| Label Spreading (KNN, inductive) | 0.01 | 0.01 |
| DistilRoBERTa | 0.89 | 0.86 |

## Evaluation Protocol

Graph-based semi-supervised models were evaluated using a **strict inductive protocol**:

- 30/70 split of labeled and unlabeled data
- TF-IDF fitted only on training + unlabeled pool
- predictions made using `.predict()` on unseen data

This prevents **transductive leakage**, where test samples influence the propagation graph and artificially inflate performance.

## Repository Structure

data/  
Dataset documentation and preprocessing.

ssl_pipeline.py  
Core semi-supervised learning pipeline.

run_experiment.py  
Runs experiments and produces evaluation metrics.

requirements.txt  
Python dependencies required to run the experiments.

ssl_emotion_classification.pdf.pdf  
Research manuscript associated with this project.

## Reproducibility Note

The final manuscript reports graph-based SSL results under a strict inductive evaluation protocol.  
If you wish to reproduce the final paper results, use the updated graph-based evaluation script included in this repository.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main SSL experiment:

```bash
python run_experiment.py
```

Run the strict inductive graph-based evaluation:

```bash
python ssl_graph_inductive_evaluation.py
```

## Citation

If you use this code or dataset in your research, please cite:

Iftikhar, H. (2026) Semi-Supervised Emotion Classification under Limited Labeled Data: A Comparative Study of Classical SSL Methods and Transformer Models. Manuscript under journal submission.

### BibTeX

@article{iftikhar2026ssl, title={Semi-Supervised Emotion Classification under Limited Labeled Data: A Comparative Study of Classical SSL Methods and Transformer Models}, author={Iftikhar, Hasan}, year={2026}, note={Manuscript under journal submission} }
