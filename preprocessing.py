# import os
# import joblib
# import pandas as pd
# import logging

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# from config import *

# os.makedirs(LOG_DIR, exist_ok=True)

# logging.basicConfig(
#     filename=os.path.join(LOG_DIR, "preprocessing.log"),
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s"
# )

# logger = logging.getLogger(__name__)

# def main():

#     print("=" * 60)
#     print("CREDIT CARD FRAUD DATA PREPROCESSING")
#     print("=" * 60)

#     # -------------------------------------------------
#     # Load Dataset
#     # -------------------------------------------------

#     print("\nLoading Dataset...")

#     df = pd.read_csv(RAW_DATA)

#     logger.info("Loading Dataset...")
#     print("Loading Dataset...")

#     # -------------------------------------------------
#     # Dataset Information
#     # -------------------------------------------------

#     print("\nDataset Shape")
#     print(df.shape)

#     print("\nColumns")
#     print(df.columns.tolist())

#     print("\nData Types")
#     print(df.dtypes)

#     # -------------------------------------------------
#     # Missing Values
#     # -------------------------------------------------

#     print("\nChecking Missing Values")

#     missing = df.isnull().sum()

#     print(missing)

#     if missing.sum() == 0:
#         print("\nNo Missing Values Found")
#     else:
#         print("\nMissing Values Detected")

#     # -------------------------------------------------
#     # Duplicate Check
#     # -------------------------------------------------

#     duplicates = df.duplicated().sum()

#     print("\nDuplicate Rows :", duplicates)

#     if duplicates > 0:

#         df = df.drop_duplicates()

#         print("Duplicate Rows Removed")

#     # -------------------------------------------------
#     # Class Distribution
#     # -------------------------------------------------

#     print("\nClass Distribution")

#     print(df["Class"].value_counts())

#     print("\nClass Percentage")

#     print(df["Class"].value_counts(normalize=True) * 100)

#     # -------------------------------------------------
#     # Feature Target Split
#     # -------------------------------------------------

#     X = df.drop(columns=["Class"])

#     y = df["Class"]

#     # -------------------------------------------------
#     # Feature Scaling
#     # -------------------------------------------------

#     print("\nScaling Time and Amount...")

#     scaler = StandardScaler()

#     X[["Time", "Amount"]] = scaler.fit_transform(
#         X[["Time", "Amount"]]
#     )

#     # -------------------------------------------------
#     # Train Test Split
#     # -------------------------------------------------

#     print("\nSplitting Dataset...")

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=TEST_SIZE,
#         random_state=RANDOM_STATE,
#         stratify=y
#     )

#     X_train, X_valid, y_train, y_valid = train_test_split(
#         X_train,
#         y_train,
#         test_size=0.125,
#         random_state=RANDOM_STATE,
#         stratify=y_train
#     )

#     # -------------------------------------------------
#     # Create Processed Folder
#     # -------------------------------------------------

#     os.makedirs(PROCESSED_DATA, exist_ok=True)

#     # -------------------------------------------------
#     # Save Processed Files
#     # -------------------------------------------------

#     print("\nSaving Processed Files...")

#     X_train.to_csv(
#         os.path.join(PROCESSED_DATA, "X_train.csv"),
#         index=False
#     )

#     X_valid.to_csv(
#         os.path.join(PROCESSED_DATA, "X_valid.csv"),
#         index=False
#     )

#     X_test.to_csv(
#         os.path.join(PROCESSED_DATA, "X_test.csv"),
#         index=False
#     )

#     y_train.to_csv(
#         os.path.join(PROCESSED_DATA, "y_train.csv"),
#         index=False
#     )

#     y_valid.to_csv(
#         os.path.join(PROCESSED_DATA, "y_valid.csv"),
#         index=False
#     )

#     y_test.to_csv(
#         os.path.join(PROCESSED_DATA, "y_test.csv"),
#         index=False
#     )

#     joblib.dump(
#         scaler,
#         os.path.join(PROCESSED_DATA, "scaler.pkl")
#     )

#     # -------------------------------------------------
#     # Summary
#     # -------------------------------------------------

#     print("\n" + "=" * 60)
#     print("PREPROCESSING COMPLETED SUCCESSFULLY")
#     print("=" * 60)

#     print(f"\nTraining Samples   : {len(X_train)}")
#     print(f"Validation Samples : {len(X_valid)}")
#     print(f"Testing Samples    : {len(X_test)}")

#     print("\nProcessed Files Saved In:")

#     print(PROCESSED_DATA)

#     print("\nScaler Saved Successfully")


# if __name__ == "__main__":
#     main()

import os
import logging
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import *

# ==========================================================
# Create Logs Directory
# ==========================================================

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "preprocessing.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# Main Function
# ==========================================================

def main():

    try:

        print("=" * 70)
        print("CREDIT CARD FRAUD DATA PREPROCESSING")
        print("=" * 70)

        logger.info("=" * 70)
        logger.info("PREPROCESSING STARTED")
        logger.info("=" * 70)

        # ==================================================
        # Load Dataset
        # ==================================================

        logger.info("Loading Dataset")

        df = pd.read_csv(RAW_DATA)

        logger.info("Dataset Loaded Successfully")

        print("\nDataset Loaded Successfully")

        # ==================================================
        # Dataset Summary
        # ==================================================

        print("\nDataset Shape :", df.shape)

        logger.info(f"Dataset Shape : {df.shape}")

        print("\nColumns")

        print(df.columns.tolist())

        logger.info(f"Columns : {df.columns.tolist()}")

        print("\nData Types")

        print(df.dtypes)

        logger.info("Data Types Verified")

        # ==================================================
        # Missing Values
        # ==================================================

        print("\nChecking Missing Values...")

        missing = df.isnull().sum()

        print(missing)

        total_missing = missing.sum()

        logger.info(f"Total Missing Values : {total_missing}")

        if total_missing == 0:

            print("\nNo Missing Values Found")

            logger.info("No Missing Values Found")

        else:

            logger.warning("Missing Values Detected")

        # ==================================================
        # Duplicate Rows
        # ==================================================

        duplicates = df.duplicated().sum()

        print(f"\nDuplicate Rows : {duplicates}")

        logger.info(f"Duplicate Rows : {duplicates}")

        if duplicates > 0:

            df.drop_duplicates(inplace=True)

            logger.info("Duplicate Rows Removed")

            print("Duplicate Rows Removed")

        # ==================================================
        # Class Distribution
        # ==================================================

        print("\nClass Distribution")

        print(df["Class"].value_counts())

        logger.info(
            f"\nClass Distribution\n{df['Class'].value_counts()}"
        )

        print("\nClass Percentage")

        print(df["Class"].value_counts(normalize=True) * 100)

        # ==================================================
        # Feature Target Split
        # ==================================================

        X = df.drop(columns=["Class"])

        y = df["Class"]

        logger.info("Feature Target Split Completed")

        # ==================================================
        # Feature Scaling
        # ==================================================

        print("\nScaling Time and Amount...")

        scaler = StandardScaler()

        X[["Time", "Amount"]] = scaler.fit_transform(
            X[["Time", "Amount"]]
        )

        logger.info("Feature Scaling Completed")

        # ==================================================
        # Train Test Split
        # ==================================================

        print("\nSplitting Dataset...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=0.125,
            random_state=RANDOM_STATE,
            stratify=y_train
        )

        logger.info("Train Validation Test Split Completed")

        # ==================================================
        # Create Processed Folder
        # ==================================================

        os.makedirs(PROCESSED_DATA, exist_ok=True)

        # ==================================================
        # Save Files
        # ==================================================

        print("\nSaving Processed Files...")

        X_train.to_csv(
            os.path.join(PROCESSED_DATA, "X_train.csv"),
            index=False
        )

        X_valid.to_csv(
            os.path.join(PROCESSED_DATA, "X_valid.csv"),
            index=False
        )

        X_test.to_csv(
            os.path.join(PROCESSED_DATA, "X_test.csv"),
            index=False
        )

        y_train.to_csv(
            os.path.join(PROCESSED_DATA, "y_train.csv"),
            index=False
        )

        y_valid.to_csv(
            os.path.join(PROCESSED_DATA, "y_valid.csv"),
            index=False
        )

        y_test.to_csv(
            os.path.join(PROCESSED_DATA, "y_test.csv"),
            index=False
        )

        joblib.dump(
            scaler,
            os.path.join(PROCESSED_DATA, "scaler.pkl")
        )

        logger.info("Processed Files Saved Successfully")

        # ==================================================
        # Final Summary
        # ==================================================

        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(f"\nTraining Samples   : {len(X_train)}")
        print(f"Validation Samples : {len(X_valid)}")
        print(f"Testing Samples    : {len(X_test)}")

        logger.info(f"Training Samples : {len(X_train)}")
        logger.info(f"Validation Samples : {len(X_valid)}")
        logger.info(f"Testing Samples : {len(X_test)}")

        logger.info("Scaler Saved Successfully")

        logger.info("=" * 70)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception as e:

        logger.exception("Error occurred during preprocessing")

        print("\nERROR OCCURRED\n")

        print(e)


# ==========================================================
# Run Program
# ==========================================================

if __name__ == "__main__":
    main()