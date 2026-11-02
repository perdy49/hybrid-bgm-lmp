import numpy as np
import pandas as pd

from services.config import (
    TARGET_COLUMN,
    ID_COLUMN,
    TIME_COLUMN,
)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional fraud detection features.

    Notes:
        - Does not modify the original DataFrame.
        - Does not use the target column to construct features.
        - Does not calculate statistics across the full dataset
          that could leak validation information.
    """
    df = df.copy()

    # =====================================================
    # 1. TRANSACTION AMOUNT
    # =====================================================

    if "TransactionAmt" in df.columns:
        amount = pd.to_numeric(
            df["TransactionAmt"],
            errors="coerce",
        )

        # Log transformation reduces extreme skew.
        df["amount_log"] = np.log1p(amount.clip(lower=0))

        # Negative amounts are retained in the original column.
        # The transformed feature is intended for nonnegative amounts.

    # =====================================================
    # 2. TRANSACTION TIME
    # =====================================================

    if TIME_COLUMN in df.columns:
        transaction_time = pd.to_numeric(
            df[TIME_COLUMN],
            errors="coerce",
        )

        # IEEE-CIS TransactionDT is relative time, not a real timestamp.
        # Use relative elapsed-time features instead of claiming
        # these are actual clock hours.
        df["transaction_time_log"] = np.log1p(transaction_time.clip(lower=0))

        df["transaction_time_bucket"] = transaction_time // 86400

    # =====================================================
    # 3. EMAIL DOMAIN MATCH
    # =====================================================

    if {"P_emaildomain", "R_emaildomain"}.issubset(df.columns):
        p_email = df["P_emaildomain"].astype("string").str.strip().str.lower()

        r_email = df["R_emaildomain"].astype("string").str.strip().str.lower()

        valid_email = (
            p_email.notna()
            & r_email.notna()
            & p_email.ne("")
            & r_email.ne("")
            & p_email.ne("nan")
            & r_email.ne("nan")
        )

        df["email_match"] = np.where(
            valid_email,
            (p_email.eq(r_email).fillna(False)).astype("int8"),
            np.nan,
        )

    # =====================================================
    # 4. CARD FEATURES
    # =====================================================

    card_columns = [
        column
        for column in [
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
        ]
        if column in df.columns
    ]

    if card_columns:
        # Count how many card fields are missing.
        df["card_missing_count"] = df[card_columns].isna().sum(axis=1)

    # =====================================================
    # 5. ADDRESS MATCH
    # =====================================================

    if {"addr1", "addr2"}.issubset(df.columns):
        addr1 = df["addr1"]
        addr2 = df["addr2"]

        valid_address = addr1.notna() & addr2.notna()

        df["addr_match"] = np.where(
            valid_address,
            (addr1 == addr2).astype("int8"),
            np.nan,
        )

    # =====================================================
    # 6. MISSING VALUES
    # =====================================================

    # Count missing values before the later preprocessing stage.
    # Exclude target and transaction identifier.
    columns_to_exclude = {
        TARGET_COLUMN,
        ID_COLUMN,
    }

    feature_columns = [
        column for column in df.columns if column not in columns_to_exclude
    ]

    df["missing_count"] = df[feature_columns].isna().sum(axis=1)

    # =====================================================
    # 7. AGGREGATE FEATURES
    # =====================================================

    # These features are created only when the source columns exist.
    # Avoid summing arbitrary groups of columns.
    for group_name, columns in {
        "C_sum": [f"C{i}" for i in range(1, 15)],
        "D_sum": [f"D{i}" for i in range(1, 16)],
    }.items():
        available_columns = [column for column in columns if column in df.columns]

        if available_columns:
            numeric_values = df[available_columns].apply(
                pd.to_numeric,
                errors="coerce",
            )

            df[group_name] = numeric_values.sum(
                axis=1,
                min_count=1,
            )

    v_columns = [
        column
        for column in df.columns
        if column.startswith("V") and column[1:].isdigit()
    ]

    if v_columns:
        numeric_v = df[v_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        df["V_mean"] = numeric_v.mean(axis=1)

    # =====================================================
    # 8. CLEAN INFINITE VALUES
    # =====================================================

    numeric_columns = df.select_dtypes(include=[np.number]).columns

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df
