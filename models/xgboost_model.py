import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np

# ==========================================================
# Add Project Root to Python Path
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# Imports
# ==========================================================

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

from config import PROCESSED_DATA, MODEL_PATH


# ==========================================================
# Directories
# ==========================================================

LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "logs"
)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)


# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    filename=os.path.join(
        LOG_DIR,
        "xgboost.log"
    ),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# Load Processed Data
# ==========================================================

def load_data():

    print("\nLoading processed datasets...")

    logger.info("Loading processed datasets")

    X_train = pd.read_csv(
        os.path.join(
            PROCESSED_DATA,
            "X_train.csv"
        )
    )

    X_valid = pd.read_csv(
        os.path.join(
            PROCESSED_DATA,
            "X_valid.csv"
        )
    )

    X_test = pd.read_csv(
        os.path.join(
            PROCESSED_DATA,
            "X_test.csv"
        )
    )

    y_train = pd.read_csv(
        os.path.join(
            PROCESSED_DATA,
            "y_train.csv"
        )
    ).squeeze("columns")

    y_valid = pd.read_csv(
        os.path.join(
            PROCESSED_DATA,
            "y_valid.csv"
        )
    ).squeeze("columns")

    y_test = pd.read_csv(
        os.path.join(
            PROCESSED_DATA,
            "y_test.csv"
        )
    ).squeeze("columns")

    print("Data loaded successfully.")

    print(
        f"Training samples   : {len(X_train):,}"
    )

    print(
        f"Validation samples : {len(X_valid):,}"
    )

    print(
        f"Testing samples    : {len(X_test):,}"
    )

    logger.info(
        f"Train={len(X_train)}, "
        f"Validation={len(X_valid)}, "
        f"Test={len(X_test)}"
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    )


# ==========================================================
# Build XGBoost Model
# ==========================================================

def build_model():

    model = XGBClassifier(

        n_estimators=500,

        learning_rate=0.05,

        max_depth=8,

        min_child_weight=2,

        subsample=0.85,

        colsample_bytree=0.85,

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        random_state=42,

        n_jobs=-1,

        verbosity=1
    )

    return model


# ==========================================================
# Train Model
# ==========================================================

def train_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid
):

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    logger.info("XGBoost training started")

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (X_train, y_train),
            (X_valid, y_valid)
        ],

        verbose=50
    )

    print("\nTraining completed.")

    logger.info("XGBoost training completed")

    return model


# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("XGBOOST RESULTS")
    print("=" * 70)

    # ------------------------------------------------------
    # Fraud Probability
    # ------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # ------------------------------------------------------
    # Classification Threshold
    # ------------------------------------------------------

    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    # ------------------------------------------------------
    # Print Metrics
    # ------------------------------------------------------

    print(
        f"Accuracy  : {accuracy:.6f}"
    )

    print(
        f"Precision : {precision:.6f}"
    )

    print(
        f"Recall    : {recall:.6f}"
    )

    print(
        f"F1        : {f1:.6f}"
    )

    print(
        f"ROC AUC   : {roc_auc:.6f}"
    )

    print(
        f"PR AUC    : {pr_auc:.6f}"
    )

    # ------------------------------------------------------
    # Classification Report
    # ------------------------------------------------------

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            predictions,
            digits=6,
            zero_division=0
        )
    )

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------

    print("Confusion Matrix")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------

    logger.info(
        f"Accuracy={accuracy:.6f}, "
        f"Precision={precision:.6f}, "
        f"Recall={recall:.6f}, "
        f"F1={f1:.6f}, "
        f"ROC_AUC={roc_auc:.6f}, "
        f"PR_AUC={pr_auc:.6f}"
    )

    return probabilities, predictions


# ==========================================================
# Save Model Predictions
# ==========================================================

def save_predictions(
    probabilities,
    predictions,
    y_test
):

    confidence = np.maximum(
        probabilities,
        1 - probabilities
    )

    prediction_data = pd.DataFrame({

        "actual": y_test.values,

        "xgb_probability": probabilities,

        "xgb_prediction": predictions,

        "xgb_confidence": confidence

    })

    output_path = os.path.join(
        PROCESSED_DATA,
        "xgb_test_predictions.csv"
    )

    prediction_data.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nPredictions saved to:\n{output_path}"
    )

    logger.info(
        f"Predictions saved to {output_path}"
    )


# ==========================================================
# Save Model
# ==========================================================

def save_model(model):

    model_path = os.path.join(
        MODEL_PATH,
        "xgboost_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nModel saved to:\n{model_path}"
    )

    logger.info(
        f"Model saved to {model_path}"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    try:

        print("=" * 70)
        print("XGBOOST FRAUD DETECTION MODEL")
        print("=" * 70)

        logger.info("=" * 70)
        logger.info("XGBOOST PIPELINE STARTED")
        logger.info("=" * 70)

        # --------------------------------------------------
        # Load Data
        # --------------------------------------------------

        (
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test
        ) = load_data()

        # --------------------------------------------------
        # Build Model
        # --------------------------------------------------

        print("\nBuilding XGBoost model...")

        model = build_model()

        logger.info(
            "XGBoost model created"
        )

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        model = train_model(
            model,
            X_train,
            y_train,
            X_valid,
            y_valid
        )

        # --------------------------------------------------
        # Evaluate
        # --------------------------------------------------

        probabilities, predictions = evaluate_model(
            model,
            X_test,
            y_test
        )

        # --------------------------------------------------
        # Save Predictions
        # --------------------------------------------------

        save_predictions(
            probabilities,
            predictions,
            y_test
        )

        # --------------------------------------------------
        # Save Model
        # --------------------------------------------------

        save_model(model)

        # --------------------------------------------------
        # Final Message
        # --------------------------------------------------

        print("\n" + "=" * 70)
        print("XGBOOST PHASE COMPLETED SUCCESSFULLY")
        print("=" * 70)

        logger.info(
            "XGBoost pipeline completed successfully"
        )

    except Exception as e:

        logger.exception(
            "Error occurred during XGBoost pipeline"
        )

        print("\nERROR OCCURRED:")
        print(e)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()