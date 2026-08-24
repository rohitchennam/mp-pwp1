import os
import sys

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# IMPORTANT:
# IMPORT PYTORCH BEFORE NUMPY / PANDAS
# ==========================================================

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==========================================================
# OTHER IMPORTS
# ==========================================================

import logging
import numpy as np
import pandas as pd

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
# CONFIGURATION
# ==========================================================

RANDOM_STATE = 42

BATCH_SIZE = 1024

EPOCHS = 30

LEARNING_RATE = 0.001

PATIENCE = 5


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")


# ==========================================================
# DIRECTORIES
# ==========================================================

LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "logs"
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_PATH,
    exist_ok=True
)


# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    filename=os.path.join(
        LOG_DIR,
        "mlp.log"
    ),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data():

    print("\nLoading processed datasets...")

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
        f"Training samples: {len(X_train)}"
    )

    logger.info(
        f"Validation samples: {len(X_valid)}"
    )

    logger.info(
        f"Testing samples: {len(X_test)}"
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
# CREATE DATALOADER
# ==========================================================

def create_dataloader(
    X,
    y,
    shuffle=False
):

    X_tensor = torch.tensor(
        X.values,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y.values,
        dtype=torch.float32
    ).reshape(-1, 1)

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

    return loader


# ==========================================================
# MLP MODEL
# ==========================================================

class FraudMLP(nn.Module):

    def __init__(
        self,
        input_size
    ):

        super().__init__()

        self.network = nn.Sequential(

            # Layer 1
            nn.Linear(
                input_size,
                128
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            # Layer 2
            nn.Linear(
                128,
                64
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            # Layer 3
            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            # Output
            nn.Linear(
                32,
                1
            )
        )

    def forward(self, x):

        return self.network(x)


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(
    model,
    train_loader,
    valid_loader
):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_valid_loss = float("inf")

    patience_counter = 0

    best_model_state = None

    print("\n")
    print("=" * 70)
    print("TRAINING MLP")
    print("=" * 70)

    for epoch in range(
        EPOCHS
    ):

        # ==================================================
        # TRAINING
        # ==================================================

        model.train()

        train_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            outputs = model(
                X_batch
            )

            loss = criterion(
                outputs,
                y_batch
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * len(X_batch)
            )

        train_loss /= len(
            train_loader.dataset
        )

        # ==================================================
        # VALIDATION
        # ==================================================

        model.eval()

        valid_loss = 0.0

        with torch.no_grad():

            for X_batch, y_batch in valid_loader:

                X_batch = X_batch.to(
                    DEVICE
                )

                y_batch = y_batch.to(
                    DEVICE
                )

                outputs = model(
                    X_batch
                )

                loss = criterion(
                    outputs,
                    y_batch
                )

                valid_loss += (
                    loss.item()
                    * len(X_batch)
                )

        valid_loss /= len(
            valid_loader.dataset
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {valid_loss:.6f}"
        )

        logger.info(
            f"Epoch={epoch + 1}, "
            f"TrainLoss={train_loss:.6f}, "
            f"ValidLoss={valid_loss:.6f}"
        )

        # ==================================================
        # EARLY STOPPING
        # ==================================================

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            patience_counter = 0

            best_model_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "\nEarly stopping triggered."
                )

                break

    # ======================================================
    # RESTORE BEST MODEL
    # ======================================================

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

    model.to(
        DEVICE
    )

    print(
        "\nMLP training completed."
    )

    return model


# ==========================================================
# PREDICTION
# ==========================================================

def predict_probabilities(
    model,
    X
):

    model.eval()

    X_tensor = torch.tensor(
        X.values,
        dtype=torch.float32
    ).to(
        DEVICE
    )

    probabilities = []

    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            BATCH_SIZE
        ):

            batch = X_tensor[
                start:start + BATCH_SIZE
            ]

            logits = model(
                batch
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.extend(
                probs.cpu()
                .numpy()
                .flatten()
            )

    return np.array(
        probabilities
    )


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n")
    print("=" * 70)
    print("MLP RESULTS")
    print("=" * 70)

    probabilities = predict_probabilities(
        model,
        X_test
    )

    # Standard fraud threshold
    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # ======================================================
    # METRICS
    # ======================================================

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

    # ======================================================
    # DISPLAY
    # ======================================================

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

    print(
        "\nClassification Report"
    )

    print(
        classification_report(
            y_test,
            predictions,
            digits=6,
            zero_division=0
        )
    )

    print(
        "Confusion Matrix"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    logger.info(
        f"Accuracy={accuracy:.6f}"
    )

    logger.info(
        f"Precision={precision:.6f}"
    )

    logger.info(
        f"Recall={recall:.6f}"
    )

    logger.info(
        f"F1={f1:.6f}"
    )

    logger.info(
        f"ROC_AUC={roc_auc:.6f}"
    )

    logger.info(
        f"PR_AUC={pr_auc:.6f}"
    )

    return (
        probabilities,
        predictions
    )


# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

def save_predictions(
    probabilities,
    predictions,
    y_test
):

    # Confidence is the probability
    # of the selected class

    confidence = np.maximum(
        probabilities,
        1 - probabilities
    )

    results = pd.DataFrame({

        "actual":
            y_test.values,

        "mlp_probability":
            probabilities,

        "mlp_prediction":
            predictions,

        "mlp_confidence":
            confidence
    })

    output_path = os.path.join(
        PROCESSED_DATA,
        "mlp_test_predictions.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(
        "\nPredictions saved to:"
    )

    print(
        output_path
    )

    logger.info(
        f"Predictions saved to {output_path}"
    )


# ==========================================================
# SAVE MODEL
# ==========================================================

def save_model(
    model,
    input_size
):

    model_path = os.path.join(
        MODEL_PATH,
        "mlp_model.pt"
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "input_size":
            input_size,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "epochs":
            EPOCHS,

        "random_state":
            RANDOM_STATE
    }

    torch.save(
        checkpoint,
        model_path
    )

    print(
        "\nModel saved to:"
    )

    print(
        model_path
    )

    logger.info(
        f"Model saved to {model_path}"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    try:

        print("=" * 70)
        print("MLP CREDIT CARD FRAUD DETECTION")
        print("=" * 70)

        logger.info(
            "MLP pipeline started"
        )

        # ==================================================
        # LOAD DATA
        # ==================================================

        (
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test
        ) = load_data()

        # ==================================================
        # CREATE LOADERS
        # ==================================================

        train_loader = create_dataloader(
            X_train,
            y_train,
            shuffle=True
        )

        valid_loader = create_dataloader(
            X_valid,
            y_valid,
            shuffle=False
        )

        # ==================================================
        # CREATE MODEL
        # ==================================================

        input_size = X_train.shape[1]

        print(
            f"\nInput Features : {input_size}"
        )

        model = FraudMLP(
            input_size
        ).to(
            DEVICE
        )

        print(
            "\nMLP Architecture:"
        )

        print(model)

        # ==================================================
        # TRAIN
        # ==================================================

        model = train_model(
            model,
            train_loader,
            valid_loader
        )

        # ==================================================
        # EVALUATE
        # ==================================================

        (
            probabilities,
            predictions
        ) = evaluate_model(
            model,
            X_test,
            y_test
        )

        # ==================================================
        # SAVE PREDICTIONS
        # ==================================================

        save_predictions(
            probabilities,
            predictions,
            y_test
        )

        # ==================================================
        # SAVE MODEL
        # ==================================================

        save_model(
            model,
            input_size
        )

        # ==================================================
        # COMPLETE
        # ==================================================

        print("\n")
        print("=" * 70)
        print("MLP PHASE COMPLETED SUCCESSFULLY")
        print("=" * 70)

        logger.info(
            "MLP pipeline completed successfully"
        )

    except Exception as e:

        logger.exception(
            "MLP pipeline failed"
        )

        print(
            "\nERROR OCCURRED:"
        )

        print(e)

        raise


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()