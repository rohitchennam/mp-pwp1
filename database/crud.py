import pandas as pd
from database.connection import engine


def insert_transactions(df: pd.DataFrame, batch_size: int = 5000):

    if df.empty:
        print("No transactions to insert.")
        return

    # ==========================================================
    # Normalize Column Names
    # ==========================================================

    # Remove spaces and convert all column names to lowercase
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # Convert target column name
    if "class" in df.columns:
        df = df.rename(
            columns={"class": "actual_class"}
        )

    # ==========================================================
    # Expected Columns
    # ==========================================================

    expected_columns = [
        "time",
        "amount",

        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v8",
        "v9",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",

        "actual_class"
    ]

    # ==========================================================
    # Check Missing Columns
    # ==========================================================

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:

        print("\nMissing Columns:")
        print(missing_columns)

        print("\nActual Dataset Columns:")
        print(df.columns.tolist())

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # ==========================================================
    # Select Required Columns
    # ==========================================================

    df = df[expected_columns]

    print(
        f"\nPreparing to insert "
        f"{len(df):,} transactions..."
    )

    # ==========================================================
    # Insert Data in Batches
    # ==========================================================

    for start in range(
        0,
        len(df),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(df)
        )

        batch = df.iloc[start:end]

        batch.to_sql(
            "transactions",
            con=engine,
            if_exists="append",
            index=False,
            method="multi"
        )

        print(
            f"Inserted "
            f"{end:,} / {len(df):,} transactions"
        )

    print(
        "\nAll transactions inserted successfully."
    )