import numpy as np
import sys
import traceback

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from services.train import train_models
from services.evaluate import (
    evaluate_models,
    save_training_history,
)
from services.config import DEFAULT_THRESHOLD


def optimize_hybrid(y, lgb, mlp):
    best = None

    for weight in np.arange(0, 1.01, 0.05):
        probabilities = weight * lgb + (1 - weight) * mlp

        for threshold in np.arange(0.05, 0.96, 0.01):
            pred = (probabilities >= threshold).astype(int)

            precision = precision_score(y, pred, zero_division=0)
            recall = recall_score(y, pred, zero_division=0)
            f1 = f1_score(y, pred, zero_division=0)

            # Jaga recall agar tidak terlalu rendah.
            if recall < 0.70:
                continue

            result = {
                "weight": float(weight),
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

            if best is None or f1 > best["f1"]:
                best = result

    if best is None:
        raise ValueError("Optimasi hybrid gagal.")

    return best


def main():
    print("=" * 70)
    print("HYBRID LIGHTGBM + MLP OPTIMIZATION")
    print("=" * 70)

    try:
        result = train_models()

        y = result["y_validation"]
        lgb = result["lightgbm_probabilities"]
        mlp = result["mlp_probabilities"]

        # Pisahkan data tuning dan evaluasi.
        indices = np.arange(len(y))

        tune_idx, test_idx = train_test_split(
            indices,
            test_size=0.50,
            random_state=42,
            stratify=y,
        )

        best = optimize_hybrid(
            y[tune_idx],
            lgb[tune_idx],
            mlp[tune_idx],
        )

        print("\nOPTIMASI HYBRID")
        print("-" * 50)
        print(f"LightGBM weight : {best['weight']:.2f}")
        print(f"MLP weight      : {1-best['weight']:.2f}")
        print(f"Threshold       : {best['threshold']:.4f}")
        print(f"Tuning Precision: {best['precision']:.4f}")
        print(f"Tuning Recall   : {best['recall']:.4f}")
        print(f"Tuning F1       : {best['f1']:.4f}")

        # Terapkan parameter hasil tuning pada holdout.
        hybrid_test = (
            best["weight"] * lgb[test_idx] + (1 - best["weight"]) * mlp[test_idx]
        )

        model_probabilities = {
            "LightGBM": lgb[test_idx],
            "MLP": mlp[test_idx],
            "Hybrid LightGBM + MLP": hybrid_test,
        }

        thresholds = {
            "LightGBM": DEFAULT_THRESHOLD,
            "MLP": DEFAULT_THRESHOLD,
            "Hybrid LightGBM + MLP": best["threshold"],
        }

        evaluate_models(
            y[test_idx],
            model_probabilities,
            threshold=thresholds,
        )

        save_training_history(result.get("history"))

        print("\nEvaluasi selesai.")

    except Exception as error:
        print(f"\nERROR: {error}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
