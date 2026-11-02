import numpy as np
import tensorflow as tf
import joblib

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from services.config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    LIGHTGBM_PARAMS,
    MLP_HIDDEN_UNITS,
    MLP_DROPOUT_RATE,
    MLP_LEARNING_RATE,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LIGHTGBM_MODEL_PATH,
    MLP_MODEL_PATH,
    LIGHTGBM_WEIGHT,
    MLP_WEIGHT,
    DEFAULT_THRESHOLD,
    MODEL_DIR,
)
from services.data_loader import load_training_data
from services.preprocess import (
    preprocess_training_data,
    prepare_train_and_validation,
)


def build_mlp(input_dim):
    """
    Build a multilayer perceptron for binary classification.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for units in MLP_HIDDEN_UNITS:
        model.add(Dense(units, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(MLP_DROPOUT_RATE))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(
            learning_rate=MLP_LEARNING_RATE,
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy",
            ),
            tf.keras.metrics.AUC(
                name="auc",
            ),
            tf.keras.metrics.Precision(
                name="precision",
            ),
            tf.keras.metrics.Recall(
                name="recall",
            ),
        ],
    )

    return model


def calculate_class_weights(y):
    """
    Calculate class weights from training labels only.
    """
    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )

    return {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }


def combine_probabilities(
    lightgbm_probabilities,
    mlp_probabilities,
):
    """
    Combine model probabilities using a weighted average.
    """
    total_weight = LIGHTGBM_WEIGHT + MLP_WEIGHT

    if total_weight <= 0:
        raise ValueError(
            "Jumlah bobot model harus lebih besar dari nol."
        )

    combined = (
        LIGHTGBM_WEIGHT * lightgbm_probabilities
        + MLP_WEIGHT * mlp_probabilities
    ) / total_weight

    return combined


def train_models():
    """
    Train LightGBM and MLP and return validation predictions.
    """
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    print("=" * 65)
    print("LIGHTGBM + MLP HYBRID FRAUD DETECTION")
    print("=" * 65)

    # -----------------------------------------------------
    # 1. LOAD DATA
    # -----------------------------------------------------

    print("\n[1/7] Memuat dataset...")

    df = load_training_data()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Kolom target '{TARGET_COLUMN}' tidak ditemukan."
        )

    print(f"Ukuran dataset: {df.shape}")

    # -----------------------------------------------------
    # 2. FEATURE ENGINEERING
    # -----------------------------------------------------

    print("\n[2/7] Membuat fitur...")

    X, y = preprocess_training_data(df)

    del df

    print(f"Jumlah fitur: {X.shape[1]:,}")
    print(f"Jumlah transaksi: {X.shape[0]:,}")
    print(f"Non-fraud: {(y == 0).sum():,}")
    print(f"Fraud: {(y == 1).sum():,}")

    # -----------------------------------------------------
    # 3. SPLIT DATA
    # -----------------------------------------------------

    print("\n[3/7] Membagi data training dan validasi...")

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    del X, y

    print(f"Data training: {len(X_train):,}")
    print(f"Data validasi: {len(X_validation):,}")

    # -----------------------------------------------------
    # 4. PREPROCESS
    # -----------------------------------------------------

    print("\n[4/7] Melakukan preprocessing...")

    (
        X_train_processed,
        X_validation_processed,
        X_train_scaled,
        X_validation_scaled,
        preprocessors,
    ) = prepare_train_and_validation(
        X_train,
        X_validation,
    )

    del X_train
    del X_validation

    y_train_array = np.asarray(
        y_train,
        dtype=np.int32,
    )

    y_validation_array = np.asarray(
        y_validation,
        dtype=np.int32,
    )

    print("Preprocessing selesai.")

    # -----------------------------------------------------
    # 5. TRAIN LIGHTGBM
    # -----------------------------------------------------

    print("\n[5/7] Melatih LightGBM...")

    lightgbm_params = LIGHTGBM_PARAMS.copy()

    if len(np.unique(y_train_array)) != 2:
        raise ValueError(
            "Data training harus memiliki kelas fraud dan non-fraud."
        )

    if lightgbm_params.get("class_weight") is None:
        lightgbm_params["class_weight"] = "balanced"

    lightgbm_model = LGBMClassifier(
        **lightgbm_params,
    )

    lightgbm_model.fit(
        X_train_processed,
        y_train_array,
    )

    lightgbm_probabilities = lightgbm_model.predict_proba(
        X_validation_processed
    )[:, 1]

    joblib.dump(
        lightgbm_model,
        LIGHTGBM_MODEL_PATH,
    )

    print(f"LightGBM disimpan: {LIGHTGBM_MODEL_PATH}")

    # -----------------------------------------------------
    # 6. TRAIN MLP
    # -----------------------------------------------------

    print("\n[6/7] Melatih MLP...")

    mlp_model = build_mlp(
        input_dim=X_train_scaled.shape[1],
    )

    class_weights = calculate_class_weights(
        y_train_array,
    )

    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = mlp_model.fit(
        X_train_scaled,
        y_train_array,
        validation_split=0.15,
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1,
    )

    mlp_probabilities = mlp_model.predict(
        X_validation_scaled,
        batch_size=MLP_BATCH_SIZE,
        verbose=0,
    ).ravel()

    mlp_model.save(
        MLP_MODEL_PATH,
    )

    print(f"MLP disimpan: {MLP_MODEL_PATH}")

    # -----------------------------------------------------
    # 7. HYBRID PREDICTIONS
    # -----------------------------------------------------

    print("\n[7/7] Menggabungkan prediksi...")

    hybrid_probabilities = combine_probabilities(
        lightgbm_probabilities,
        mlp_probabilities,
    )

    print("\nTraining selesai.")
    print(f"LightGBM weight: {LIGHTGBM_WEIGHT}")
    print(f"MLP weight: {MLP_WEIGHT}")
    print(f"Threshold awal: {DEFAULT_THRESHOLD}")

    return {
        "y_validation": y_validation_array,
        "lightgbm_probabilities": lightgbm_probabilities,
        "mlp_probabilities": mlp_probabilities,
        "hybrid_probabilities": hybrid_probabilities,
        "history": history,
        "lightgbm_model": lightgbm_model,
        "mlp_model": mlp_model,
        "preprocessors": preprocessors,
    }
