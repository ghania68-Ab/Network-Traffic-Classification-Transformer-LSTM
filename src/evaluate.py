"""Evaluation and visualization utilities.

This module only creates experiment artifacts. It does not generate a final PDF
report automatically; the report content is maintained separately in
report/final_report_content.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str]) -> dict[str, object]:
    """Evaluate one trained model on the held-out test split."""

    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(weighted_precision),
        "recall": float(weighted_recall),
        "f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }


def _distribution_lines(class_distribution: Mapping[str, Mapping[str, int]]) -> list[str]:
    lines = ["Class distribution:"]
    for split, counts in class_distribution.items():
        rendered = ", ".join(f"{label}: {count}" for label, count in counts.items())
        lines.append(f"- {split}: {rendered}")
    return lines


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Create a readable ASCII table for metrics.txt."""

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(str(value))) for width, value in zip(widths, row)]

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_line = "|" + "|".join(f" {header:<{width}} " for header, width in zip(headers, widths)) + "|"
    table = [border, header_line, border]
    for row in rows:
        table.append("|" + "|".join(f" {str(value):<{width}} " for value, width in zip(row, widths)) + "|")
    table.append(border)
    return table


def save_metrics(experiments: Mapping[str, Mapping[str, object]], output_path: str | Path) -> None:
    """Save one metrics file for the supplied dataset experiment(s)."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    summary_headers = [
        "Dataset",
        "Model",
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1",
        "Macro F1",
    ]
    summary_rows = []
    for experiment in experiments.values():
        dataset_name = str(experiment["dataset_name"])
        for model_name, result in experiment["metrics"].items():
            summary_rows.append([
                dataset_name,
                model_name,
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1']:.4f}",
                f"{result['macro_f1']:.4f}",
            ])

    lines = [
        "Network Traffic Classification Metrics",
        "",
        "Model Comparison Summary",
        *_format_table(summary_headers, summary_rows),
    ]

    for experiment in experiments.values():
        lines.extend(["", "=" * 100, f"Dataset: {experiment['dataset_name']}"])
        lines.append(f"Label column: {experiment['label_column']}")
        lines.extend(_distribution_lines(experiment["class_distribution"]))
        for model_name, result in experiment["metrics"].items():
            metric_rows = [[
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1']:.4f}",
                f"{result['macro_f1']:.4f}",
            ]]
            lines.extend([
                "",
                f"Metrics: {model_name}",
                *_format_table(["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1", "Macro F1"], metric_rows),
                "",
                f"Classification Report: {model_name}",
                str(result["classification_report"]),
            ])

    output.write_text("\n".join(lines), encoding="utf-8")

def plot_accuracy_curve(experiments: Mapping[str, Mapping[str, object]], output_path: str | Path) -> None:
    """Save training and validation accuracy curves for supplied experiments."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.5, 6), dpi=170)
    for experiment in experiments.values():
        dataset_name = str(experiment["dataset_name"])
        for model_name, history in experiment["histories"].items():
            values = history.history
            epochs = range(1, len(values["accuracy"]) + 1)
            label_prefix = f"{dataset_name} - {model_name}"
            plt.plot(epochs, values["accuracy"], label=f"{label_prefix} train")
            plt.plot(epochs, values["val_accuracy"], linestyle="--", label=f"{label_prefix} val")

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.30)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_loss_curve(experiments: Mapping[str, Mapping[str, object]], output_path: str | Path) -> None:
    """Save training and validation loss curves for supplied experiments."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.5, 6), dpi=170)
    for experiment in experiments.values():
        dataset_name = str(experiment["dataset_name"])
        for model_name, history in experiment["histories"].items():
            values = history.history
            epochs = range(1, len(values["loss"]) + 1)
            label_prefix = f"{dataset_name} - {model_name}"
            plt.plot(epochs, values["loss"], label=f"{label_prefix} train")
            plt.plot(epochs, values["val_loss"], linestyle="--", label=f"{label_prefix} val")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.30)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(experiments: Mapping[str, Mapping[str, object]], output_path: str | Path) -> None:
    """Save normalized confusion matrices for supplied experiments."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    for experiment in experiments.values():
        for model_name, result in experiment["metrics"].items():
            panels.append((experiment, model_name, result))

    cols = 2
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5.2 * rows), dpi=160)
    axes = np.atleast_1d(axes).ravel()

    for ax, (experiment, model_name, result) in zip(axes, panels):
        matrix = np.asarray(result["confusion_matrix"])
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.zeros_like(matrix, dtype=float)
        np.divide(matrix, row_sums, out=normalized, where=row_sums != 0)
        display = ConfusionMatrixDisplay(normalized, display_labels=experiment["class_names"])
        display.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
        ax.set_title(f"{experiment['dataset_name']} - {model_name}")
        ax.tick_params(axis="x", labelrotation=45)

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle("Normalized Confusion Matrices", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_graph(experiments: Mapping[str, Mapping[str, object]], output_path: str | Path) -> None:
    """Save a compact model comparison graph."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels, accuracy, weighted_f1, macro_f1 = [], [], [], []
    for experiment in experiments.values():
        for model_name, result in experiment["metrics"].items():
            labels.append(f"{experiment['dataset_name']}\n{model_name}")
            accuracy.append(float(result["accuracy"]))
            weighted_f1.append(float(result["f1"]))
            macro_f1.append(float(result["macro_f1"]))

    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(11.5, 6), dpi=170)
    bars1 = plt.bar(x - width, accuracy, width, label="Accuracy")
    bars2 = plt.bar(x, weighted_f1, width, label="Weighted F1")
    bars3 = plt.bar(x + width, macro_f1, width, label="Macro F1")
    for bars in (bars1, bars2, bars3):
        plt.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    plt.title("Model Comparison Across Datasets")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
