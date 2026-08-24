import pandas as pd

from config import RAW_DATA
from database.crud import insert_transactions


def main():

    print("=" * 60)
    print("LOADING TRANSACTIONS INTO POSTGRESQL")
    print("=" * 60)

    print("\nReading dataset...")

    df = pd.read_csv(RAW_DATA)

    print(f"Dataset shape: {df.shape}")

    insert_transactions(df)

    print("\nDatabase loading completed.")


if __name__ == "__main__":
    main()