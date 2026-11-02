
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
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
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


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Calculate classification metrics from predicted probabilities.

    Metrics:
        Accuracy
        Precision
        Recall
        F1-score
        ROC-AUC
        PR-AUC (Average Precision)
        Specificity
        Balanced Accuracy
        MCC
        TN, FP, FN, TP
    """

    y_true = np.asarray(y_true).ravel()
    probabilities = np.asarray(probabilities).ravel()

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Jumlah y_true dan probabilities tidak sama."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Data evaluasi kosong."
        )

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError(
            "Label harus berupa 0 (Non-Fraud) atau 1 (Fraud)."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Probabilitas mengandung NaN atau infinity."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(
            "Probabilitas harus berada antara 0 dan 1."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold harus berada antara 0 dan 1."
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    # Basic Classification Metrics
    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    # ROC-AUC and PR-AUC
    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_true,
            probabilities,
        )

    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    # Specificity
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    # Balanced Accuracy
    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predictions,
    )

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(
        y_true,
        predictions,
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),

        "threshold": threshold,
        "predictions": predictions,
        "probabilities": probabilities,
    }

    return metrics


# ============================================================
# EVALUATE MODELS
# ============================================================

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

    if not model_probabilities:
        raise ValueError(
            "Tidak ada probabilitas model untuk dievaluasi."
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

        print(
            f"Threshold          : {metrics['threshold']:.4f}"
        )

        print(
            f"Accuracy           : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision          : {metrics['precision']:.4f}"
        )

        print(
            f"Recall             : {metrics['recall']:.4f}"
        )

        print(
            f"F1-score           : {metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC            : {metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC (AP)        : {metrics['pr_auc']:.4f}"
        )

        print(
            f"Specificity        : {metrics['specificity']:.4f}"
        )

        print(
            f"Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"MCC                : {metrics['mcc']:.4f}"
        )

        print("\nConfusion Matrix:")
        print("-" * 55)

        print(
            f"True Negative (TN) : {metrics['true_negative']}"
        )

        print(
            f"False Positive (FP): {metrics['false_positive']}"
        )

        print(
            f"False Negative (FN): {metrics['false_negative']}"
        )

        print(
            f"True Positive (TP) : {metrics['true_positive']}"
        )

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


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

def find_best_threshold(
    y_true,
    probabilities,
    metric="f1",
    min_precision=None,
    min_recall=None,
):
    """
    Find the best classification threshold using
    validation probabilities.

    metric:
        f1 or precision

    min_precision:
        Optional minimum precision constraint.

    min_recall:
        Optional minimum recall constraint.
    """

    y_true = np.asarray(y_true).ravel()
    probabilities = np.asarray(probabilities).ravel()

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Jumlah y_true dan probabilities tidak sama."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Data validation kosong."
        )

    if metric not in ["f1", "precision"]:
        raise ValueError(
            "Metric hanya mendukung 'f1' atau 'precision'."
        )

    if min_precision is not None and not 0 <= min_precision <= 1:
        raise ValueError(
            "min_precision harus berada antara 0 dan 1."
        )

    if min_recall is not None and not 0 <= min_recall <= 1:
        raise ValueError(
            "min_recall harus berada antara 0 dan 1."
        )

    thresholds = np.arange(
        0.05,
        0.96,
        0.01,
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if min_precision is not None and precision < min_precision:
            continue

        if min_recall is not None and recall < min_recall:
            continue

        score = (
            f1
            if metric == "f1"
            else precision
        )

        results.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "score": float(score),
            }
        )

    if not results:
        raise ValueError(
            "Tidak ditemukan threshold yang memenuhi "
            "batas precision dan recall."
        )

    best_result = max(
        results,
        key=lambda item: item["score"],
    )

    return best_result, results


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

def save_confusion_matrices(
    y_true,
    all_metrics,
):
    """
    Save confusion matrices for all models.
    """

    if not all_metrics:
        return

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


# ============================================================
# SAVE ROC CURVES
# ============================================================

def save_roc_curves(
    y_true,
    model_probabilities,
):
    """
    Save ROC curves for all models.
    """

    y_true = np.asarray(y_true).ravel()

    if len(np.unique(y_true)) != 2:
        print(
            "ROC curve tidak dibuat karena "
            "data evaluasi tidak memiliki kedua kelas."
        )
        return

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for model_name, probabilities in model_probabilities.items():

        probabilities = np.asarray(
            probabilities
        ).ravel()

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

    plt.tight_layout()

    fig.savefig(
        ROC_CURVE_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"ROC curve disimpan: {ROC_CURVE_PATH}"
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

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

    if "loss" in history_data or "val_loss" in history_data:
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

    if "auc" in history_data or "val_auc" in history_data:
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


# ============================================================
# SAVE METRICS REPORT
# ============================================================

def save_metrics_report(all_metrics):
    """
    Save complete evaluation metrics to a text file.
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
                "-" * 60,
                f"Threshold: {metrics['threshold']:.4f}",
                "",
                "Classification Metrics",
                f"Accuracy: {metrics['accuracy']:.6f}",
                f"Precision: {metrics['precision']:.6f}",
                f"Recall: {metrics['recall']:.6f}",
                f"F1-score: {metrics['f1']:.6f}",
                f"ROC-AUC: {metrics['roc_auc']:.6f}",
                f"PR-AUC (Average Precision): {metrics['pr_auc']:.6f}",
                f"Specificity: {metrics['specificity']:.6f}",
                f"Balanced Accuracy: {metrics['balanced_accuracy']:.6f}",
                f"MCC: {metrics['mcc']:.6f}",
                "",
                "Confusion Matrix",
                f"True Negative (TN): {metrics['true_negative']}",
                f"False Positive (FP): {metrics['false_positive']}",
                f"False Negative (FN): {metrics['false_negative']}",
                f"True Positive (TP): {metrics['true_positive']}",
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
