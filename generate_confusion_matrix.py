#!/usr/bin/env python3
"""
Confusion matrix generator for the noMNIST experiments.
Requires: numpy, pandas, scikit-learn, matplotlib, seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

PREDICTIONS_PATH = "test_results/nomnist_predictions.csv"
DATASET_PATH = "dat/test_nomnist.dat"
CLASS_NAMES = ["a", "b", "c", "d", "e", "f"]


def load_predictions(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if "Category" not in df.columns:
        raise ValueError("CSV must contain a 'Category' column with predicted labels.")
    return df["Category"].to_numpy()


def load_true_labels(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) != 3:
            raise ValueError("First line of dataset must contain: n_inputs n_outputs n_patterns")
        _n_inputs, n_outputs, n_patterns = map(int, header)

        data = np.loadtxt(f)
        if data.shape[0] != n_patterns:
            raise ValueError("Number of patterns in file does not match header")
        if data.shape[1] < n_outputs:
            raise ValueError("Dataset rows must end with the one-hot encoded targets")

    one_hot = data[:, -n_outputs:]
    return one_hot.argmax(axis=1)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str = "test_results/confusion_matrix.png") -> None:
    """
    Create a visual heatmap of the confusion matrix.
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        save_path: Path to save the figure
    """
    # Normalize confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create figure with two subplots: raw counts and percentages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    ax1.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Class', fontsize=14, fontweight='bold')
    ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=16, fontweight='bold')
    
    # Plot 2: Normalized percentages
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2,
        cbar_kws={'label': 'Percentage'},
        linewidths=0.5,
        linecolor='gray',
        vmin=0,
        vmax=1
    )
    ax2.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Class', fontsize=14, fontweight='bold')
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to {save_path}")
    plt.close()


def main() -> None:
    y_pred = load_predictions(PREDICTIONS_PATH)
    y_true = load_true_labels(DATASET_PATH)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Predicted and true label arrays must have the same length")

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)

    print("=" * 72)
    print("CONFUSION MATRIX for noMNIST (rows=true, cols=predicted)")
    print("=" * 72)
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))
    print()
    print(report)

    np.savetxt(
        "test_results/confusion_matrix.txt",
        cm,
        fmt="%d",
        header="Confusion Matrix (rows=true, cols=predicted)\nClasses: a=0, b=1, c=2, d=3, e=4, f=5",
        comments="",
    )
    np.savetxt("test_results/confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    print("Confusion matrix written to test_results/confusion_matrix.txt/.csv")
    
    # Generate visualization
    plot_confusion_matrix(cm, CLASS_NAMES)


if __name__ == "__main__":
    main()
