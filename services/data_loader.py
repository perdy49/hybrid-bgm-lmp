import pandas as pd

from services.config import (
    TRAIN_TRANSACTION_PATH,
    TRAIN_IDENTITY_PATH,
    TEST_TRANSACTION_PATH,
    TEST_IDENTITY_PATH,
    TARGET_COLUMN,
    ID_COLUMN,
)


def load_csv(path):
    """
    Load a CSV file and raise a clear error if it is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}\n"
            "Pastikan file CSV berada di folder data/."
        )

    print(f"[LOAD] Membaca: {path.name}")

    df = pd.read_csv(path, low_memory=False)

    print(
        f"[INFO] {path.name}: "
        f"{df.shape[0]:,} baris, {df.shape[1]:,} kolom"
    )

    return df


def merge_transaction_identity(
    transaction_df,
    identity_df,
):
    """
    Merge transaction data with identity data using TransactionID.
    """
    if identity_df is None:
        return transaction_df

    if ID_COLUMN not in transaction_df.columns:
        raise ValueError(
            f"Kolom '{ID_COLUMN}' tidak ditemukan "
            "di dataset transaksi."
        )

    if ID_COLUMN not in identity_df.columns:
        raise ValueError(
            f"Kolom '{ID_COLUMN}' tidak ditemukan "
            "di dataset identitas."
        )

    if identity_df[ID_COLUMN].duplicated().any():
        raise ValueError(
            f"Dataset identitas memiliki TransactionID duplikat. "
            "Periksa data sebelum melakukan merge."
        )

    print("[INFO] Menggabungkan data transaksi dan identitas...")

    merged_df = transaction_df.merge(
        identity_df,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    print(
        f"[INFO] Hasil merge: "
        f"{merged_df.shape[0]:,} baris, "
        f"{merged_df.shape[1]:,} kolom"
    )

    return merged_df


def load_training_data():
    """
    Load and merge IEEE-CIS training transaction and identity data.

    Returns:
        pandas.DataFrame: Merged training dataset.
    """
    transaction_df = load_csv(TRAIN_TRANSACTION_PATH)
    identity_df = load_csv(TRAIN_IDENTITY_PATH)

    if TARGET_COLUMN not in transaction_df.columns:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' tidak ditemukan "
            "di dataset training."
        )

    train_df = merge_transaction_identity(
        transaction_df,
        identity_df,
    )

    # The target must contain binary labels 0 and 1.
    if train_df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Target '{TARGET_COLUMN}' mengandung nilai kosong."
        )

    unique_targets = set(train_df[TARGET_COLUMN].unique())

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target harus berupa label biner 0 dan 1. "
            f"Label yang ditemukan: {unique_targets}"
        )

    print("\n[INFO] Distribusi target training:")
    print(train_df[TARGET_COLUMN].value_counts().sort_index())

    print("\n[INFO] Persentase target:")
    print(
        (
            train_df[TARGET_COLUMN]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    return train_df


def load_test_data():
    """
    Load and merge IEEE-CIS test transaction and identity data.

    Note:
        The official IEEE-CIS test dataset generally does not
        contain the isFraud target. It cannot be used for
        supervised evaluation unless labels are available separately.
    """
    transaction_df = load_csv(TEST_TRANSACTION_PATH)
    identity_df = load_csv(TEST_IDENTITY_PATH)

    test_df = merge_transaction_identity(
        transaction_df,
        identity_df,
    )

    if TARGET_COLUMN in test_df.columns:
        print(
            f"[INFO] Kolom '{TARGET_COLUMN}' ditemukan "
            "pada dataset test."
        )
    else:
        print(
            f"[INFO] Kolom '{TARGET_COLUMN}' tidak ditemukan "
            "pada dataset test. Dataset ini tidak dapat digunakan "
            "untuk evaluasi supervised tanpa label yang tersedia."
        )

    return test_df


def print_dataset_summary(df, name="Dataset"):
    """
    Print a compact summary of a dataset.
    """
    print(f"\n========== {name.upper()} ==========")
    print(f"Jumlah baris : {len(df):,}")
    print(f"Jumlah kolom : {len(df.columns):,}")
    print(f"Penggunaan memori: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nContoh data:")
    print(df.head(3))

    print("\nJumlah nilai kosong terbanyak:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("Tidak ada nilai kosong.")
    else:
        print(missing.head(10))
