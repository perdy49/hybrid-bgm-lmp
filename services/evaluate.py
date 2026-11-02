import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

from services.config import (
    RESULTS_DIR,
    CONFUSION_MATRIX_PATH,
    ROC_CURVE_PATH,
    TRAINING_HISTORY_PATH,
    METRICS_PATH,
    DEFAULT_THRESHOLD,
)


def calculate_metrics(
    y_true,
    probabilities,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Calculate classification metrics from predicted probabilities.
    """
    probabilities = np.asarray(
        probabilities
    ).ravel()

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": (
            roc_auc_score(
                y_true,
                probabilities,
            )
            if len(np.unique(y_true)) == 2
            else float("nan")
        ),
        "threshold": threshold,
        "predictions": predictions,
        "probabilities": probabilities,
    }

    return metrics


def evaluate_models(
    y_true,
    model_probabilities,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Evaluate LightGBM, MLP, and hybrid probabilities.
    """
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics = {}

    print("\n" + "=" * 75)
    print("HASIL EVALUASI MODEL FRAUD DETECTION")
    print("=" * 75)

    for model_name, probabilities in model_probabilities.items():
        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold=threshold,
        )

        all_metrics[model_name] = metrics

        print(f"\nMODEL: {model_name.upper()}")
        print("-" * 55)
        print(f"Threshold : {metrics['threshold']:.4f}")
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1-score  : {metrics['f1']:.4f}")
        print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_true,
                metrics["predictions"],
                labels=[0, 1],
                target_names=[
                    "Non-Fraud",
                    "Fraud",
                ],
                zero_division=0,
            )
        )

    save_confusion_matrices(
        y_true,
        all_metrics,
    )

    save_roc_curves(
        y_true,
        model_probabilities,
    )

    save_metrics_report(
        all_metrics,
    )

    return all_metrics


def save_confusion_matrices(
    y_true,
    all_metrics,
):
    """
    Save confusion matrices for all models.
    """
    model_names = list(
        all_metrics.keys()
    )

    fig, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(6 * len(model_names), 5),
    )

    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(
        axes,
        model_names,
    ):
        predictions = all_metrics[
            model_name
        ]["predictions"]

        matrix = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "Non-Fraud",
                "Fraud",
            ],
            yticklabels=[
                "Non-Fraud",
                "Fraud",
            ],
            ax=ax,
        )

        ax.set_title(
            model_name
        )
        ax.set_xlabel(
            "Prediksi"
        )
        ax.set_ylabel(
            "Aktual"
        )

    plt.tight_layout()

    fig.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nConfusion matrix disimpan: "
        f"{CONFUSION_MATRIX_PATH}"
    )


def save_roc_curves(
    y_true,
    model_probabilities,
):
    """
    Save ROC curves for all models.
    """
    if len(np.unique(y_true)) != 2:
        print(
            "ROC curve tidak dibuat karena "
            "data evaluasi tidak memiliki kedua kelas."
        )
        return

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for model_name, probabilities in (
        model_probabilities.items()
    ):
        fpr, tpr, _ = roc_curve(
            y_true,
            probabilities,
        )

        auc = roc_auc_score(
            y_true,
            probabilities,
        )

        ax.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC={auc:.4f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax.set_title(
        "ROC Curve - Fraud Detection"
    )
    ax.set_xlabel(
        "False Positive Rate"
    )
    ax.set_ylabel(
        "True Positive Rate"
    )
    ax.legend()
    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        ROC_CURVE_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"ROC curve disimpan: {ROC_CURVE_PATH}"
    )


def save_training_history(history):
    """
    Save the MLP training and validation loss/metrics plot.
    """
    if history is None:
        return

    history_data = history.history

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    if "loss" in history_data:
        axes[0].plot(
            history_data["loss"],
            label="Training Loss",
        )

    if "val_loss" in history_data:
        axes[0].plot(
            history_data["val_loss"],
            label="Validation Loss",
        )

    axes[0].set_title(
        "MLP Loss"
    )
    axes[0].set_xlabel(
        "Epoch"
    )
    axes[0].set_ylabel(
        "Loss"
    )
    axes[0].legend()

    if "auc" in history_data:
        axes[1].plot(
            history_data["auc"],
            label="Training AUC",
        )

    if "val_auc" in history_data:
        axes[1].plot(
            history_data["val_auc"],
            label="Validation AUC",
        )

    axes[1].set_title(
        "MLP AUC"
    )
    axes[1].set_xlabel(
        "Epoch"
    )
    axes[1].set_ylabel(
        "AUC"
    )
    axes[1].legend()

    plt.tight_layout()

    fig.savefig(
        TRAINING_HISTORY_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Training history disimpan: "
        f"{TRAINING_HISTORY_PATH}"
    )


def save_metrics_report(all_metrics):
    """
    Save evaluation metrics to a text file.
    """
    lines = [
        "LIGHTGBM + MLP FRAUD DETECTION",
        "=" * 60,
        "",
    ]

    for model_name, metrics in all_metrics.items():
        lines.extend(
            [
                f"MODEL: {model_name.upper()}",
                f"Threshold: {metrics['threshold']:.4f}",
                f"Accuracy: {metrics['accuracy']:.6f}",
                f"Precision: {metrics['precision']:.6f}",
                f"Recall: {metrics['recall']:.6f}",
                f"F1-score: {metrics['f1']:.6f}",
                f"ROC-AUC: {metrics['roc_auc']:.6f}",
                "",
            ]
        )

    METRICS_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"Metrics disimpan: {METRICS_PATH}"
    )