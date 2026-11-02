import joblib
import tensorflow as tf

from services.config import (
    LIGHTGBM_MODEL_PATH,
    MLP_MODEL_PATH,
    SCALER_PATH,
    FEATURE_NAMES_PATH,
    LABEL_ENCODERS_PATH,
)


def save_preprocessors(preprocessors):
    """
    Save preprocessing objects used during training.
    """
    joblib.dump(
        preprocessors,
        LABEL_ENCODERS_PATH,
    )

    joblib.dump(
        preprocessors["scaler"],
        SCALER_PATH,
    )

    joblib.dump(
        list(
            preprocessors["numerical_columns"]
            + preprocessors["categorical_columns"]
        ),
        FEATURE_NAMES_PATH,
    )


def load_preprocessors():
    """
    Load preprocessing objects saved during training.
    """
    if not LABEL_ENCODERS_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor tidak ditemukan: {LABEL_ENCODERS_PATH}"
        )

    return joblib.load(LABEL_ENCODERS_PATH)


def load_hybrid_models():
    """
    Load the trained LightGBM and MLP models.

    Returns:
        dict: Both models and preprocessing objects.
    """
    missing_files = [
        path
        for path in (
            LIGHTGBM_MODEL_PATH,
            MLP_MODEL_PATH,
            SCALER_PATH,
            FEATURE_NAMES_PATH,
            LABEL_ENCODERS_PATH,
        )
        if not path.exists()
    ]

    if missing_files:
        missing = "\n".join(
            str(path) for path in missing_files
        )

        raise FileNotFoundError(
            "Beberapa file model belum tersedia:\n"
            f"{missing}\n"
            "Jalankan training terlebih dahulu."
        )

    lightgbm_model = joblib.load(
        LIGHTGBM_MODEL_PATH
    )

    mlp_model = tf.keras.models.load_model(
        MLP_MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    feature_names = joblib.load(
        FEATURE_NAMES_PATH
    )

    preprocessors = joblib.load(
        LABEL_ENCODERS_PATH
    )

    return {
        "lightgbm_model": lightgbm_model,
        "mlp_model": mlp_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "preprocessors": preprocessors,
    }
