import sys
import traceback

from services.train import train_models
from services.evaluate import (
    evaluate_models,
    save_training_history,
)
from services.config import DEFAULT_THRESHOLD


def main():
    print("=" * 75)
    print("SISTEM DETEKSI FRAUD - LIGHTGBM + MLP")
    print("=" * 75)

    try:
        # 1. Train models.
        training_result = train_models()

        # 2. Prepare probabilities for evaluation.
        model_probabilities = {
            "LightGBM": training_result[
                "lightgbm_probabilities"
            ],
            "MLP": training_result[
                "mlp_probabilities"
            ],
            "Hybrid LightGBM + MLP": training_result[
                "hybrid_probabilities"
            ],
        }

        # 3. Evaluate models.
        metrics = evaluate_models(
            y_true=training_result[
                "y_validation"
            ],
            model_probabilities=model_probabilities,
            threshold=DEFAULT_THRESHOLD,
        )

        # 4. Save MLP training history.
        save_training_history(
            training_result.get("history")
        )

        # 5. Print comparison.
        print("\n" + "=" * 75)
        print("PERBANDINGAN AKHIR")
        print("=" * 75)

        print(
            f"{'Model':<25}"
            f"{'Accuracy':>12}"
            f"{'Precision':>12}"
            f"{'Recall':>12}"
            f"{'F1-score':>12}"
            f"{'ROC-AUC':>12}"
        )

        for model_name, result in metrics.items():
            print(
                f"{model_name:<25}"
                f"{result['accuracy']:>12.4f}"
                f"{result['precision']:>12.4f}"
                f"{result['recall']:>12.4f}"
                f"{result['f1']:>12.4f}"
                f"{result['roc_auc']:>12.4f}"
            )

        print("\nTraining dan evaluasi selesai.")
        print("Periksa folder results/ untuk grafik dan laporan.")

    except FileNotFoundError as error:
        print(f"\n[FILE ERROR] {error}")
        sys.exit(1)

    except Exception as error:
        print(f"\n[ERROR] {error}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
