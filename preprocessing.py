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


#2 version
# import os
# import logging
# import joblib
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# from config import *

# # ==========================================================
# # Create Logs Directory
# # ==========================================================

# os.makedirs(LOG_DIR, exist_ok=True)

# logging.basicConfig(
#     filename=os.path.join(LOG_DIR, "preprocessing.log"),
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s"
# )

# logger = logging.getLogger(__name__)


# # ==========================================================
# # Main Function
# # ==========================================================

# def main():

#     try:

#         print("=" * 70)
#         print("CREDIT CARD FRAUD DATA PREPROCESSING")
#         print("=" * 70)

#         logger.info("=" * 70)
#         logger.info("PREPROCESSING STARTED")
#         logger.info("=" * 70)

#         # ==================================================
#         # Load Dataset
#         # ==================================================

#         logger.info("Loading Dataset")

#         df = pd.read_csv(RAW_DATA)

#         logger.info("Dataset Loaded Successfully")

#         print("\nDataset Loaded Successfully")

#         # ==================================================
#         # Dataset Summary
#         # ==================================================

#         print("\nDataset Shape :", df.shape)

#         logger.info(f"Dataset Shape : {df.shape}")

#         print("\nColumns")

#         print(df.columns.tolist())

#         logger.info(f"Columns : {df.columns.tolist()}")

#         print("\nData Types")

#         print(df.dtypes)

#         logger.info("Data Types Verified")

#         # ==================================================
#         # Missing Values
#         # ==================================================

#         print("\nChecking Missing Values...")

#         missing = df.isnull().sum()

#         print(missing)

#         total_missing = missing.sum()

#         logger.info(f"Total Missing Values : {total_missing}")

#         if total_missing == 0:

#             print("\nNo Missing Values Found")

#             logger.info("No Missing Values Found")

#         else:

#             logger.warning("Missing Values Detected")

#         # ==================================================
#         # Duplicate Rows
#         # ==================================================

#         duplicates = df.duplicated().sum()

#         print(f"\nDuplicate Rows : {duplicates}")

#         logger.info(f"Duplicate Rows : {duplicates}")

#         if duplicates > 0:

#             df.drop_duplicates(inplace=True)

#             logger.info("Duplicate Rows Removed")

#             print("Duplicate Rows Removed")

#         # ==================================================
#         # Class Distribution
#         # ==================================================

#         print("\nClass Distribution")

#         print(df["Class"].value_counts())

#         logger.info(
#             f"\nClass Distribution\n{df['Class'].value_counts()}"
#         )

#         print("\nClass Percentage")

#         print(df["Class"].value_counts(normalize=True) * 100)

#         # ==================================================
#         # Feature Target Split
#         # ==================================================

#         X = df.drop(columns=["Class"])

#         y = df["Class"]

#         logger.info("Feature Target Split Completed")

#         # ==================================================
#         # Feature Scaling
#         # ==================================================

#         print("\nScaling Time and Amount...")

#         scaler = StandardScaler()

#         X[["Time", "Amount"]] = scaler.fit_transform(
#             X[["Time", "Amount"]]
#         )

#         logger.info("Feature Scaling Completed")

#         # ==================================================
#         # Train Test Split
#         # ==================================================

#         print("\nSplitting Dataset...")

#         X_train, X_test, y_train, y_test = train_test_split(
#             X,
#             y,
#             test_size=TEST_SIZE,
#             random_state=RANDOM_STATE,
#             stratify=y
#         )

#         X_train, X_valid, y_train, y_valid = train_test_split(
#             X_train,
#             y_train,
#             test_size=0.125,
#             random_state=RANDOM_STATE,
#             stratify=y_train
#         )

#         logger.info("Train Validation Test Split Completed")

#         # ==================================================
#         # Create Processed Folder
#         # ==================================================

#         os.makedirs(PROCESSED_DATA, exist_ok=True)

#         # ==================================================
#         # Save Files
#         # ==================================================

#         print("\nSaving Processed Files...")

#         X_train.to_csv(
#             os.path.join(PROCESSED_DATA, "X_train.csv"),
#             index=False
#         )

#         X_valid.to_csv(
#             os.path.join(PROCESSED_DATA, "X_valid.csv"),
#             index=False
#         )

#         X_test.to_csv(
#             os.path.join(PROCESSED_DATA, "X_test.csv"),
#             index=False
#         )

#         y_train.to_csv(
#             os.path.join(PROCESSED_DATA, "y_train.csv"),
#             index=False
#         )

#         y_valid.to_csv(
#             os.path.join(PROCESSED_DATA, "y_valid.csv"),
#             index=False
#         )

#         y_test.to_csv(
#             os.path.join(PROCESSED_DATA, "y_test.csv"),
#             index=False
#         )

#         joblib.dump(
#             scaler,
#             os.path.join(PROCESSED_DATA, "scaler.pkl")
#         )

#         logger.info("Processed Files Saved Successfully")

#         # ==================================================
#         # Final Summary
#         # ==================================================

#         print("\n" + "=" * 70)
#         print("PREPROCESSING COMPLETED SUCCESSFULLY")
#         print("=" * 70)

#         print(f"\nTraining Samples   : {len(X_train)}")
#         print(f"Validation Samples : {len(X_valid)}")
#         print(f"Testing Samples    : {len(X_test)}")

#         logger.info(f"Training Samples : {len(X_train)}")
#         logger.info(f"Validation Samples : {len(X_valid)}")
#         logger.info(f"Testing Samples : {len(X_test)}")

#         logger.info("Scaler Saved Successfully")

#         logger.info("=" * 70)
#         logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
#         logger.info("=" * 70)

#     except Exception as e:

#         logger.exception("Error occurred during preprocessing")

#         print("\nERROR OCCURRED\n")

#         print(e)


# # ==========================================================
# # Run Program
# # ==========================================================

# if __name__ == "__main__":
#     main()

#3 version
import os
import logging
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import *


# ==========================================================
# Logging Setup
# ==========================================================

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(
        LOG_DIR,
        "preprocessing.log"
    ),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# Main Preprocessing Pipeline
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

        print("\nLoading Dataset...")

        df = pd.read_csv(RAW_DATA)

        print("Dataset Loaded Successfully")
        print(f"Dataset Shape : {df.shape}")

        logger.info(
            f"Dataset loaded successfully: {df.shape}"
        )

        # ==================================================
        # Validate Target Column
        # ==================================================

        if "Class" not in df.columns:

            raise ValueError(
                "Target column 'Class' not found."
            )

        # ==================================================
        # Missing Values
        # ==================================================

        print("\nChecking Missing Values...")

        missing = df.isnull().sum()

        total_missing = missing.sum()

        print(
            f"Total Missing Values : {total_missing}"
        )

        logger.info(
            f"Total Missing Values: {total_missing}"
        )

        if total_missing > 0:

            print("\nMissing values detected:")
            print(
                missing[missing > 0]
            )

            raise ValueError(
                "Dataset contains missing values."
            )

        # ==================================================
        # Duplicate Rows
        # ==================================================

        duplicates = df.duplicated().sum()

        print(
            f"\nDuplicate Rows : {duplicates}"
        )

        logger.info(
            f"Duplicate Rows: {duplicates}"
        )

        if duplicates > 0:

            df = df.drop_duplicates()

            print(
                f"After removing duplicates: {df.shape}"
            )

            logger.info(
                "Duplicate rows removed."
            )

        # ==================================================
        # Class Distribution
        # ==================================================

        print("\nClass Distribution")

        class_counts = df["Class"].value_counts()

        print(class_counts)

        print("\nClass Percentage")

        class_percentage = (
            df["Class"]
            .value_counts(normalize=True)
            * 100
        )

        print(class_percentage)

        logger.info(
            f"Class Distribution:\n{class_counts}"
        )

        # ==================================================
        # Separate Features and Target
        # ==================================================

        X = df.drop(
            columns=["Class"]
        )

        y = df["Class"]

        logger.info(
            "Features and target separated."
        )

        # ==================================================
        # FIRST SPLIT
        #
        # Train = 70%
        # Temp  = 30%
        # ==================================================

        print("\nCreating Train/Test Split...")

        X_train, X_temp, y_train, y_temp = train_test_split(

            X,
            y,

            test_size=0.30,

            random_state=RANDOM_STATE,

            stratify=y
        )

        # ==================================================
        # SECOND SPLIT
        #
        # Validation = 10%
        # Test       = 20%
        #
        # Temp = 30%
        #
        # 1/3 of Temp = 10% total
        # 2/3 of Temp = 20% total
        # ==================================================

        X_valid, X_test, y_valid, y_test = train_test_split(

            X_temp,
            y_temp,

            test_size=(2 / 3),

            random_state=RANDOM_STATE,

            stratify=y_temp
        )

        print("\nDataset Split Completed")

        print(
            f"Training Samples   : {len(X_train):,}"
        )

        print(
            f"Validation Samples : {len(X_valid):,}"
        )

        print(
            f"Testing Samples    : {len(X_test):,}"
        )

        logger.info(
            f"Train: {len(X_train)}"
        )

        logger.info(
            f"Validation: {len(X_valid)}"
        )

        logger.info(
            f"Test: {len(X_test)}"
        )

        # ==================================================
        # IMPORTANT:
        # FIT SCALER ONLY ON TRAINING DATA
        # ==================================================

        print(
            "\nFitting StandardScaler ONLY on training data..."
        )

        scaler = StandardScaler()

        # Fit only on training data
        scaler.fit(
            X_train[["Time", "Amount"]]
        )

        # ==================================================
        # Transform Training Data
        # ==================================================

        X_train = X_train.copy()

        X_train[
            ["Time", "Amount"]
        ] = scaler.transform(
            X_train[["Time", "Amount"]]
        )

        # ==================================================
        # Transform Validation Data
        # ==================================================

        X_valid = X_valid.copy()

        X_valid[
            ["Time", "Amount"]
        ] = scaler.transform(
            X_valid[["Time", "Amount"]]
        )

        # ==================================================
        # Transform Test Data
        # ==================================================

        X_test = X_test.copy()

        X_test[
            ["Time", "Amount"]
        ] = scaler.transform(
            X_test[["Time", "Amount"]]
        )

        print(
            "Scaling completed without data leakage."
        )

        logger.info(
            "Scaler fitted only on training data."
        )

        # ==================================================
        # Create Processed Directory
        # ==================================================

        os.makedirs(
            PROCESSED_DATA,
            exist_ok=True
        )

        # ==================================================
        # Save X Data
        # ==================================================

        print("\nSaving processed datasets...")

        X_train.to_csv(
            os.path.join(
                PROCESSED_DATA,
                "X_train.csv"
            ),
            index=False
        )

        X_valid.to_csv(
            os.path.join(
                PROCESSED_DATA,
                "X_valid.csv"
            ),
            index=False
        )

        X_test.to_csv(
            os.path.join(
                PROCESSED_DATA,
                "X_test.csv"
            ),
            index=False
        )

        # ==================================================
        # Save Y Data
        # ==================================================

        y_train.to_csv(
            os.path.join(
                PROCESSED_DATA,
                "y_train.csv"
            ),
            index=False
        )

        y_valid.to_csv(
            os.path.join(
                PROCESSED_DATA,
                "y_valid.csv"
            ),
            index=False
        )

        y_test.to_csv(
            os.path.join(
                PROCESSED_DATA,
                "y_test.csv"
            ),
            index=False
        )

        # ==================================================
        # Save Scaler
        # ==================================================

        scaler_path = os.path.join(
            PROCESSED_DATA,
            "scaler.pkl"
        )

        joblib.dump(
            scaler,
            scaler_path
        )

        # ==================================================
        # Save Dataset Information
        # ==================================================

        split_info = pd.DataFrame({

            "dataset": [
                "train",
                "validation",
                "test"
            ],

            "samples": [
                len(X_train),
                len(X_valid),
                len(X_test)
            ],

            "fraud_cases": [
                int(y_train.sum()),
                int(y_valid.sum()),
                int(y_test.sum())
            ],

            "non_fraud_cases": [
                int((y_train == 0).sum()),
                int((y_valid == 0).sum()),
                int((y_test == 0).sum())
            ]
        })

        split_info.to_csv(
            os.path.join(
                PROCESSED_DATA,
                "split_info.csv"
            ),
            index=False
        )

        # ==================================================
        # Final Summary
        # ==================================================

        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(
            f"\nTraining Samples   : {len(X_train):,}"
        )

        print(
            f"Validation Samples : {len(X_valid):,}"
        )

        print(
            f"Testing Samples    : {len(X_test):,}"
        )

        print(
            f"\nScaler saved to:\n{scaler_path}"
        )

        print(
            "\nNo data leakage from validation/test sets."
        )

        logger.info(
            "Preprocessing completed successfully."
        )

        logger.info(
            "Scaler fitted exclusively on training data."
        )

    except Exception as e:

        logger.exception(
            "Preprocessing failed."
        )

        print("\nERROR OCCURRED:")
        print(e)

        raise


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()