import numpy as np
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from services.config import (
    TARGET_COLUMN,
    ID_COLUMN,
    MODEL_DIR,
    SCALER_PATH,
    FEATURE_NAMES_PATH,
    LABEL_ENCODERS_PATH,
)
from services.feature_engineering import create_features


def split_features_target(df):
    """
    Separate features from the target column.

    Returns:
        X: Feature DataFrame.
        y: Target Series.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Kolom target '{TARGET_COLUMN}' tidak ditemukan."
        )

    y = df[TARGET_COLUMN].astype("int8").copy()

    X = df.drop(
        columns=[TARGET_COLUMN],
        errors="ignore",
    ).copy()

    # TransactionID identifies a transaction but is not itself
    # a behavioral fraud feature.
    X = X.drop(
        columns=[ID_COLUMN],
        errors="ignore",
    )

    return X, y


def separate_column_types(X):
    """
    Identify categorical and numerical columns.
    """
    categorical_columns = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    numerical_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    return numerical_columns, categorical_columns


def clean_numeric_features(X, numerical_columns):
    """
    Convert numerical columns to numeric values and replace
    infinite values with missing values.
    """
    X = X.copy()

    for column in numerical_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X[numerical_columns] = X[numerical_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return X


def fit_preprocessor(X_train):
    """
    Fit imputers, categorical encoders, and scaler on training data.

    This function must only receive the training split.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train = X_train.copy()

    numerical_columns, categorical_columns = (
        separate_column_types(X_train)
    )

    X_train = clean_numeric_features(
        X_train,
        numerical_columns,
    )

    # Fit numerical imputer using training data only.
    numerical_imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    X_train[numerical_columns] = numerical_imputer.fit_transform(
        X_train[numerical_columns]
    )

    # Convert categorical values to strings while preserving
    # missing values as a dedicated category.
    if categorical_columns:
        X_train[categorical_columns] = (
            X_train[categorical_columns]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        X_train[categorical_columns] = (
            categorical_encoder.fit_transform(
                X_train[categorical_columns]
            )
        )
    else:
        categorical_encoder = None

    # Ensure all model features are numerical.
    X_train = X_train.astype("float32")

    # Fit scaler for the MLP input.
    scaler = StandardScaler()

    scaler.fit(X_train)

    preprocessors = {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "numerical_imputer": numerical_imputer,
        "categorical_encoder": categorical_encoder,
        "scaler": scaler,
    }

    joblib.dump(
        preprocessors,
        LABEL_ENCODERS_PATH,
    )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    joblib.dump(
        list(X_train.columns),
        FEATURE_NAMES_PATH,
    )

    return X_train, preprocessors


def transform_features(X, preprocessors):
    """
    Transform data using fitted training preprocessors.

    Unknown categories are encoded as -1.
    """
    X = X.copy()

    numerical_columns = preprocessors["numerical_columns"]
    categorical_columns = preprocessors["categorical_columns"]

    # Align incoming columns to the original training schema.
    expected_columns = (
        numerical_columns + categorical_columns
    )

    X = X.reindex(
        columns=expected_columns,
    )

    X = clean_numeric_features(
        X,
        numerical_columns,
    )

    numerical_imputer = preprocessors["numerical_imputer"]

    X[numerical_columns] = numerical_imputer.transform(
        X[numerical_columns]
    )

    categorical_encoder = preprocessors["categorical_encoder"]

    if categorical_columns:
        X[categorical_columns] = (
            X[categorical_columns]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

        X[categorical_columns] = (
            categorical_encoder.transform(
                X[categorical_columns]
            )
        )

    X = X.astype("float32")

    return X


def preprocess_training_data(df):
    """
    Create engineered features and split into X and y.

    IMPORTANT:
        This function does not fit preprocessing on the entire dataset.
        Split the returned X and y before calling fit_preprocessor().
    """
    df = create_features(df)

    X, y = split_features_target(df)

    # Retain a consistent column order.
    X = X.reindex(sorted(X.columns), axis=1)

    return X, y


def prepare_train_and_validation(
    X_train,
    X_validation,
):
    """
    Fit preprocessing on training data and transform validation data.

    Returns:
        X_train_processed
        X_validation_processed
        X_train_scaled
        X_validation_scaled
        preprocessors
    """
    X_train_processed, preprocessors = fit_preprocessor(
        X_train
    )

    X_validation_processed = transform_features(
        X_validation,
        preprocessors,
    )

    # Same feature order is required for both datasets.
    feature_names = joblib.load(
        FEATURE_NAMES_PATH
    )

    X_train_processed = X_train_processed.reindex(
        columns=feature_names,
    )

    X_validation_processed = X_validation_processed.reindex(
        columns=feature_names,
    )

    scaler = preprocessors["scaler"]

    X_train_scaled = scaler.transform(
        X_train_processed
    ).astype("float32")

    X_validation_scaled = scaler.transform(
        X_validation_processed
    ).astype("float32")

    return (
        X_train_processed,
        X_validation_processed,
        X_train_scaled,
        X_validation_scaled,
        preprocessors,
    )
