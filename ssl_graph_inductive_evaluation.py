"""
Script for strict inductive evaluation of graph-based SSL methods for emotion classification.

This script loads a dataset of text and labels from a CSV file, splits into train and test,
applies TF-IDF vectorization on the training set plus unlabeled pool, and trains LabelPropagation and LabelSpreading models
with RBF and KNN kernels. It then evaluates on the held-out test set using accuracy and macro F1 metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import accuracy_score, f1_score


def run_inductive_graph_ssl(data_path, text_column='text', label_column='label',
                            test_size=0.2, unlabeled_fraction=0.5, random_state=42):
    """Run strict inductive graph-based semi-supervised learning evaluation.

    Parameters
    ----------
    data_path : str
        Path to a CSV file containing text and labels.
    text_column : str, default='text'
        Name of the text column in the dataset.
    label_column : str, default='label'
        Name of the label column in the dataset.
    test_size : float, default=0.2
        Fraction of data to use as test set.
    unlabeled_fraction : float, default=0.5
        Fraction of training samples to treat as unlabeled.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    dict
        Dictionary mapping model names to accuracy and macro F1 metrics.
    """
    # load dataset
    df = pd.read_csv(data_path)
    X = df[text_column].astype(str).tolist()
    y = df[label_column].to_numpy()

    # train-test split (inductive)
    X_train_text, X_test_text, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    # create unlabeled mask on training data
    n_train = len(y_train_full)
    rng = np.random.RandomState(random_state)
    unlabeled_mask = rng.rand(n_train) < unlabeled_fraction
    y_train = np.copy(y_train_full)
    y_train[unlabeled_mask] = -1  # mark unlabeled as -1

    # vectorize using TF-IDF on training text only (strict inductive)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    results = {}

    for kernel in ['rbf', 'knn']:
        # define parameters depending on kernel
        gamma = 20 if kernel == 'rbf' else None
        n_neighbors = 5 if kernel == 'knn' else None

        # LabelPropagation
        lp = LabelPropagation(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, max_iter=1000)
        lp.fit(X_train_vec, y_train)
        y_pred_lp = lp.predict(X_test_vec)
        results[f'Label Propagation ({kernel})'] = {
            'accuracy': accuracy_score(y_test, y_pred_lp),
            'macro_f1': f1_score(y_test, y_pred_lp, average='macro')
        }

        # LabelSpreading
        ls = LabelSpreading(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, max_iter=1000)
        ls.fit(X_train_vec, y_train)
        y_pred_ls = ls.predict(X_test_vec)
        results[f'Label Spreading ({kernel})'] = {
            'accuracy': accuracy_score(y_test, y_pred_ls),
            'macro_f1': f1_score(y_test, y_pred_ls, average='macro')
        }

    # print results
    for model_name, metrics in results.items():
        print(f"{model_name}: Accuracy = {metrics['accuracy']:.4f}, Macro F1 = {metrics['macro_f1']:.4f}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Strict inductive evaluation of graph-based SSL models.')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to CSV dataset with text and label columns.')
    parser.add_argument('--text-column', type=str, default='text',
                        help='Name of text column in the dataset.')
    parser.add_argument('--label-column', type=str, default='label',
                        help='Name of label column in the dataset.')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size fraction.')
    parser.add_argument('--unlabeled-fraction', type=float, default=0.5,
                        help='Fraction of training data to mark as unlabeled.')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed.')
    args = parser.parse_args()

    run_inductive_graph_ssl(args.data_path,
                            text_column=args.text_column,
                            label_column=args.label_column,
                            test_size=args.test_size,
                            unlabeled_fraction=args.unlabeled_fraction,
                            random_state=args.random_state)
