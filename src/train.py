"""Train and evaluate Simple LSTM and Transformer-Enhanced LSTM models.

Run examples:
    python src/train.py --dataset cic
    python src/train.py --dataset unsw
    python src/train.py --all-datasets

Outputs are not deleted automatically. Single-dataset runs save dataset-specific
files, while all-dataset runs also save the main combined project files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from evaluate import (
    evaluate_model,
    plot_accuracy_curve,
    plot_comparison_graph,
    plot_confusion_matrices,
    plot_loss_curve,
    save_metrics,
)
from model import build_simple_lstm, build_transformer_lstm, set_global_determinism
from preprocessing import prepare_dataset

MODEL_NAMES = ("Simple LSTM", "Transformer-Enhanced LSTM")
DATASET_SLUGS = {"cic": "cic_darknet2020", "unsw": "unsw_nb15"}


def build_callbacks(results_dir: Path, model_name: str, save_models: bool = False):
    """Use validation loss to stop overfitting and reduce learning rate.

    EarlyStopping means you can request 50 or more epochs safely: training will
    stop once validation loss stops improving for the configured patience.
    """

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    if save_models:
        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        callbacks.append(
            ModelCheckpoint(results_dir / f"{safe_name}.keras", monitor="val_loss", save_best_only=True)
        )
    return callbacks


def compute_weights(y_train: np.ndarray, mode: str = "sqrt") -> Dict[int, float] | None:
    """Return class weights for imbalanced datasets."""

    if mode == "none":
        return None
    classes = np.unique(y_train)
    balanced = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weights = balanced if mode == "balanced" else np.sqrt(balanced)
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


def build_model(model_name: str, input_shape: tuple[int, int], num_classes: int):
    """Build one of the two required models with fair tuned defaults."""

    if model_name == "Simple LSTM":
        return build_simple_lstm(
            input_shape=input_shape,
            num_classes=num_classes,
            lstm_units=96,
            dropout_rate=0.25,
            dense_units=96,
            learning_rate=8e-4,
        )
    if model_name == "Transformer-Enhanced LSTM":
        return build_transformer_lstm(
            input_shape=input_shape,
            num_classes=num_classes,
            lstm_units=128,
            attention_heads=4,
            dropout_rate=0.20,
            dense_units=128,
            ffn_units=256,
            learning_rate=5e-4,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def train_single_model(
    model,
    model_name: str,
    data,
    epochs: int,
    batch_size: int,
    results_dir: Path,
    class_weight_mode: str,
    save_models: bool,
) -> Tuple[object, object, Dict[str, object]]:
    """Train a model and evaluate it on the test split."""

    print("\n" + "=" * 60)
    print(f"Dataset : {data.dataset_name}")
    print(f"Model   : {model_name}")
    print(f"Epochs  : {epochs}")
    print(f"Batch   : {batch_size}")
    print("=" * 60 + "\n")

    history = model.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=build_callbacks(results_dir, model_name, save_models=save_models),
        class_weight=compute_weights(data.y_train, class_weight_mode),
        verbose=1,
    )

    print("\n" + "=" * 60)
    print(f"Finished Training : {model_name}")
    print("=" * 60 + "\n")

    return model, history, evaluate_model(model, data.X_test, data.y_test, data.class_names)


def run_experiment(
    dataset: str = "cic",
    label_column: str | None = None,
    data_dir: str | Path = "dataset",
    results_dir: str | Path = "results",
    epochs: int = 20,
    batch_size: int = 256,
    max_samples: int | None = None,
    class_weight_mode: str = "sqrt",
    save_models: bool = False,
    seed: int = 42,
) -> Dict[str, object]:
    """Run both models on one dataset and return the experiment object."""

    data = prepare_dataset(dataset=dataset, data_dir=data_dir, label_column=label_column, max_samples=max_samples)
    histories, metrics = {}, {}
    for index, model_name in enumerate(MODEL_NAMES):
        # Reset seeds before each model so the Transformer does not inherit a
        # different random state simply because the baseline trained first.
        set_global_determinism(seed + index)
        model = build_model(model_name, data.input_shape, data.num_classes)
        _, histories[model_name], metrics[model_name] = train_single_model(
            model=model,
            model_name=model_name,
            data=data,
            epochs=epochs,
            batch_size=batch_size,
            results_dir=Path(results_dir),
            class_weight_mode=class_weight_mode,
            save_models=save_models,
        )

    return {
        "dataset_key": dataset,
        "dataset_slug": DATASET_SLUGS.get(dataset, dataset),
        "dataset_name": data.dataset_name,
        "label_column": data.label_column,
        "class_names": data.class_names,
        "class_distribution": data.class_distribution,
        "histories": histories,
        "metrics": metrics,
    }


def write_outputs(
    experiments: Mapping[str, Mapping[str, object]],
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
    combined: bool = False,
) -> None:
    """Write artifacts without deleting previous runs.

    Single-dataset files are named with the dataset slug so CIC and UNSW runs do
    not overwrite each other. Combined all-dataset runs also write the required
    project filenames: metrics.txt, confusion_matrix.png, accuracy_curve.png,
    loss_curve.png, and comparison_model_graph.png.
    """

    results_path = Path(results_dir)
    figures_path = Path(figures_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    for experiment in experiments.values():
        slug = str(experiment["dataset_slug"])
        single = {slug: experiment}
        save_metrics(single, results_path / f"{slug}_metrics.txt")
        plot_confusion_matrices(single, results_path / f"{slug}_confusion_matrix.png")
        plot_accuracy_curve(single, figures_path / f"{slug}_accuracy_curve.png")
        plot_loss_curve(single, figures_path / f"{slug}_loss_curve.png")
        plot_comparison_graph(single, figures_path / f"{slug}_comparison_model_graph.png")

    if combined:
        save_metrics(experiments, results_path / "metrics.txt")
        plot_confusion_matrices(experiments, results_path / "confusion_matrix.png")
        plot_accuracy_curve(experiments, figures_path / "accuracy_curve.png")
        plot_loss_curve(experiments, figures_path / "loss_curve.png")
        plot_comparison_graph(experiments, figures_path / "comparison_model_graph.png")


def run_project(
    dataset: str = "cic",
    all_datasets: bool = False,
    label_column: str | None = None,
    data_dir: str | Path = "dataset",
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
    epochs: int = 20,
    batch_size: int = 256,
    max_samples: int | None = None,
    class_weight_mode: str = "sqrt",
    save_models: bool = False,
    seed: int = 42,
) -> Dict[str, Mapping[str, object]]:
    """Run the requested experiment and write metrics/figures."""

    set_global_determinism(seed)
    dataset_keys = ("cic", "unsw") if all_datasets else (dataset,)
    experiments = {}
    for key in dataset_keys:
        experiment = run_experiment(
            dataset=key,
            label_column=label_column if not all_datasets else None,
            data_dir=data_dir,
            results_dir=results_dir,
            epochs=epochs,
            batch_size=batch_size,
            max_samples=max_samples,
            class_weight_mode=class_weight_mode,
            save_models=save_models,
        )
        experiments[key] = experiment

    write_outputs(experiments, results_dir, figures_dir, combined=all_datasets)
    return experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cic", "unsw"], default="cic")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--class-weight-mode", choices=["none", "sqrt", "balanced"], default="sqrt")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_project(
        dataset=args.dataset,
        all_datasets=args.all_datasets,
        label_column=args.label_column,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        class_weight_mode=args.class_weight_mode,
        save_models=args.save_models,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
