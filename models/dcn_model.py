"""
======================================================================
DCN FRAUD DETECTION
PHASE 3A

Architecture:
    XGBoost
    MLP
    FT-Transformer
    DCN

Current model:
    Deep & Cross Network (DCN)

Dataset:
    259,070 training samples
    37,010 validation samples
    74,020 testing samples
    30 continuous features

Purpose:
    Train a Deep & Cross Network for fraud detection and generate
    test-set probabilities for the model disagreement engine.
======================================================================
"""

# ====================================================================
# IMPORTANT:
# Set threading/environment variables BEFORE importing torch/numpy.
# This matches the working FT-Transformer environment.
# ====================================================================

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Prevent Windows/OpenMP DLL conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ====================================================================
# PROJECT ROOT
# ====================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ====================================================================
# IMPORT TORCH FIRST
# ====================================================================

import torch

print()
print("=" * 70)
print("DCN PYTORCH ENVIRONMENT")
print("=" * 70)

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())

if torch.cuda.is_available():

    print(
        "CUDA version    :",
        torch.version.cuda
    )

    print(
        "GPU             :",
        torch.cuda.get_device_name(0)
    )

    DEVICE = torch.device("cuda")

else:

    print("GPU             : NOT AVAILABLE")

    DEVICE = torch.device("cpu")

print("Device          :", DEVICE)

# ====================================================================
# REMAINING IMPORTS
# ====================================================================

import copy
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

# ====================================================================
# CONFIG
# ====================================================================

try:

    from config import (
        PROCESSED_DATA,
        MODEL_PATH,
    )

except ImportError:

    PROCESSED_DATA = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed"
    )

    MODEL_PATH = os.path.join(
        PROJECT_ROOT,
        "saved_models"
    )

# ====================================================================
# PATHS
# ====================================================================

X_TRAIN_PATH = os.path.join(
    PROCESSED_DATA,
    "X_train.csv"
)

X_VALID_PATH = os.path.join(
    PROCESSED_DATA,
    "X_valid.csv"
)

X_TEST_PATH = os.path.join(
    PROCESSED_DATA,
    "X_test.csv"
)

Y_TRAIN_PATH = os.path.join(
    PROCESSED_DATA,
    "y_train.csv"
)

Y_VALID_PATH = os.path.join(
    PROCESSED_DATA,
    "y_valid.csv"
)

Y_TEST_PATH = os.path.join(
    PROCESSED_DATA,
    "y_test.csv"
)

PREDICTION_PATH = os.path.join(
    PROCESSED_DATA,
    "dcn_test_predictions.csv"
)

MODEL_SAVE_DIR = MODEL_PATH

MODEL_FILE = os.path.join(
    MODEL_SAVE_DIR,
    "dcn_model.pt"
)

# ====================================================================
# DCN HYPERPARAMETERS
# ====================================================================

# Number of cross layers
NUM_CROSS_LAYERS = 3

# Deep network architecture
DEEP_HIDDEN_DIMS = [256, 128, 64]

DROPOUT = 0.10

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.00001

BATCH_SIZE = 1024

EPOCHS = 15

PATIENCE = 5

RANDOM_STATE = 42

# ====================================================================
# RANDOM SEED
# ====================================================================

def set_seed(seed=RANDOM_STATE):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ====================================================================
# LOAD DATA
# ====================================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING DATA FOR DCN")
    print("=" * 70)

    X_train = pd.read_csv(
        X_TRAIN_PATH
    )

    X_valid = pd.read_csv(
        X_VALID_PATH
    )

    X_test = pd.read_csv(
        X_TEST_PATH
    )

    y_train = pd.read_csv(
        Y_TRAIN_PATH
    )

    y_valid = pd.read_csv(
        Y_VALID_PATH
    )

    y_test = pd.read_csv(
        Y_TEST_PATH
    )

    # ---------------------------------------------------------------
    # Convert target DataFrames to Series
    # ---------------------------------------------------------------

    if isinstance(y_train, pd.DataFrame):

        y_train = y_train.iloc[:, 0]

    if isinstance(y_valid, pd.DataFrame):

        y_valid = y_valid.iloc[:, 0]

    if isinstance(y_test, pd.DataFrame):

        y_test = y_test.iloc[:, 0]

    # ---------------------------------------------------------------
    # Numeric conversion
    # ---------------------------------------------------------------

    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_valid = X_valid.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # ---------------------------------------------------------------
    # Replace infinity
    # ---------------------------------------------------------------

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_valid = X_valid.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ---------------------------------------------------------------
    # Training medians
    # ---------------------------------------------------------------

    train_medians = X_train.median()

    X_train = X_train.fillna(
        train_medians
    )

    X_valid = X_valid.fillna(
        train_medians
    )

    X_test = X_test.fillna(
        train_medians
    )

    # ---------------------------------------------------------------
    # Targets
    # ---------------------------------------------------------------

    y_train = pd.to_numeric(
        y_train,
        errors="coerce"
    ).fillna(0).astype(int)

    y_valid = pd.to_numeric(
        y_valid,
        errors="coerce"
    ).fillna(0).astype(int)

    y_test = pd.to_numeric(
        y_test,
        errors="coerce"
    ).fillna(0).astype(int)

    print()
    print(
        "Training shape   :",
        X_train.shape
    )

    print(
        "Validation shape :",
        X_valid.shape
    )

    print(
        "Testing shape    :",
        X_test.shape
    )

    print()
    print(
        "Number of features:",
        X_train.shape[1]
    )

    print(
        "Training fraud   :",
        int(y_train.sum())
    )

    print(
        "Validation fraud :",
        int(y_valid.sum())
    )

    print(
        "Testing fraud    :",
        int(y_test.sum())
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    )


# ====================================================================
# DCN CROSS NETWORK
# ====================================================================

class CrossNetwork(nn.Module):

    """
    Deep & Cross Network.

    Formula:

        x_{l+1} =
            x_0 * (w_l^T x_l)
            + b_l
            + x_l

    This explicitly models feature interactions.
    """

    def __init__(
        self,
        input_dim,
        num_layers
    ):

        super().__init__()

        self.input_dim = input_dim

        self.num_layers = num_layers

        self.weights = nn.ParameterList()

        self.biases = nn.ParameterList()

        for _ in range(num_layers):

            weight = nn.Parameter(
                torch.empty(input_dim)
            )

            bias = nn.Parameter(
                torch.zeros(input_dim)
            )

            nn.init.xavier_uniform_(
                weight.unsqueeze(0)
            )

            self.weights.append(weight)

            self.biases.append(bias)

    def forward(self, x0):

        x = x0

        for weight, bias in zip(
            self.weights,
            self.biases
        ):

            # w^T x
            scalar = torch.sum(
                x * weight,
                dim=1,
                keepdim=True
            )

            # x0 * (w^T x)
            cross = x0 * scalar

            # x_{l+1}
            x = cross + bias + x

        return x


# ====================================================================
# DEEP NETWORK
# ====================================================================

class DeepNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dims,
        dropout
    ):

        super().__init__()

        layers = []

        current_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.append(
                nn.Linear(
                    current_dim,
                    hidden_dim
                )
            )

            layers.append(
                nn.BatchNorm1d(
                    hidden_dim
                )
            )

            layers.append(
                nn.ReLU()
            )

            layers.append(
                nn.Dropout(dropout)
            )

            current_dim = hidden_dim

        self.network = nn.Sequential(
            *layers
        )

        self.output_dim = current_dim

    def forward(self, x):

        return self.network(x)


# ====================================================================
# DCN MODEL
# ====================================================================

class DCNClassifier(nn.Module):

    def __init__(
        self,
        num_features,
        num_cross_layers=NUM_CROSS_LAYERS,
        deep_hidden_dims=DEEP_HIDDEN_DIMS,
        dropout=DROPOUT
    ):

        super().__init__()

        self.num_features = num_features

        # ------------------------------------------------------------
        # Cross network
        # ------------------------------------------------------------

        self.cross_network = CrossNetwork(
            input_dim=num_features,
            num_layers=num_cross_layers
        )

        # ------------------------------------------------------------
        # Deep network
        # ------------------------------------------------------------

        self.deep_network = DeepNetwork(
            input_dim=num_features,
            hidden_dims=deep_hidden_dims,
            dropout=dropout
        )

        # ------------------------------------------------------------
        # Combine Cross + Deep
        # ------------------------------------------------------------

        combined_dim = (
            num_features
            +
            self.deep_network.output_dim
        )

        self.output_layer = nn.Linear(
            combined_dim,
            1
        )

    def forward(self, x):

        cross_output = self.cross_network(
            x
        )

        deep_output = self.deep_network(
            x
        )

        combined = torch.cat(
            [
                cross_output,
                deep_output
            ],
            dim=1
        )

        logits = self.output_layer(
            combined
        )

        return logits.squeeze(1)


# ====================================================================
# CREATE MODEL
# ====================================================================

def create_model(num_features):

    print()
    print("=" * 70)
    print("INITIALIZING DCN")
    print("=" * 70)

    print()
    print(
        "PyTorch version :",
        torch.__version__
    )

    print(
        "CUDA available  :",
        torch.cuda.is_available()
    )

    print(
        "Device          :",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU             :",
            torch.cuda.get_device_name(0)
        )

    print(
        "Input features  :",
        num_features
    )

    print(
        "Cross layers    :",
        NUM_CROSS_LAYERS
    )

    print(
        "Deep layers     :",
        DEEP_HIDDEN_DIMS
    )

    print(
        "Learning rate   :",
        LEARNING_RATE
    )

    model = DCNClassifier(
        num_features=num_features,
        num_cross_layers=NUM_CROSS_LAYERS,
        deep_hidden_dims=DEEP_HIDDEN_DIMS,
        dropout=DROPOUT
    )

    model = model.to(DEVICE)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "DCN initialized successfully"
    )

    print(
        "Total parameters    :",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    return model


# ====================================================================
# DATA LOADERS
# ====================================================================

def create_dataloaders(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test
):

    print()
    print("=" * 70)
    print("CREATING DATA LOADERS")
    print("=" * 70)

    X_train_tensor = torch.tensor(
        X_train.values,
        dtype=torch.float32
    )

    X_valid_tensor = torch.tensor(
        X_valid.values,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test.values,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32
    )

    y_valid_tensor = torch.tensor(
        y_valid.values,
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test.values,
        dtype=torch.float32
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    valid_dataset = TensorDataset(
        X_valid_tensor,
        y_valid_tensor
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print()
    print(
        "Batch size        :",
        BATCH_SIZE
    )

    print(
        "Training batches  :",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(valid_loader)
    )

    print(
        "Testing batches   :",
        len(test_loader)
    )

    return (
        train_loader,
        valid_loader,
        test_loader
    )


# ====================================================================
# EVALUATION
# ====================================================================

def evaluate(
    model,
    loader,
    criterion=None
):

    model.eval()

    all_probabilities = []

    all_targets = []

    total_loss = 0.0

    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE,
                non_blocking=True
            )

            y_batch = y_batch.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(
                X_batch
            )

            if criterion is not None:

                loss = criterion(
                    logits,
                    y_batch
                )

                batch_size = X_batch.size(0)

                total_loss += (
                    loss.item()
                    *
                    batch_size
                )

                total_samples += batch_size

            probabilities = torch.sigmoid(
                logits
            )

            all_probabilities.extend(
                probabilities.detach()
                .cpu()
                .numpy()
            )

            all_targets.extend(
                y_batch.detach()
                .cpu()
                .numpy()
            )

    probabilities = np.asarray(
        all_probabilities
    )

    targets = np.asarray(
        all_targets
    ).astype(int)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    average_loss = (
        total_loss / total_samples
        if total_samples > 0
        else 0.0
    )

    return (
        average_loss,
        targets,
        probabilities,
        predictions
    )


# ====================================================================
# TRAIN
# ====================================================================

def train_model(
    model,
    train_loader,
    valid_loader
):

    print()
    print("=" * 70)
    print("STARTING DCN TRAINING")
    print("=" * 70)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_auc = -np.inf

    best_state = None

    patience_counter = 0

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        total_samples = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                DEVICE,
                non_blocking=True
            )

            y_batch = y_batch.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_size = X_batch.size(0)

            running_loss += (
                loss.item()
                *
                batch_size
            )

            total_samples += batch_size

        train_loss = (
            running_loss
            /
            total_samples
        )

        (
            val_loss,
            y_true,
            y_probability,
            y_prediction
        ) = evaluate(
            model,
            valid_loader,
            criterion
        )

        val_auc = roc_auc_score(
            y_true,
            y_probability
        )

        val_f1 = f1_score(
            y_true,
            y_prediction,
            zero_division=0
        )

        val_precision = precision_score(
            y_true,
            y_prediction,
            zero_division=0
        )

        val_recall = recall_score(
            y_true,
            y_prediction,
            zero_division=0
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.6f} | "
            f"Val F1: {val_f1:.6f} | "
            f"Val Precision: {val_precision:.6f} | "
            f"Val Recall: {val_recall:.6f}"
        )

        # ------------------------------------------------------------
        # BEST MODEL
        # ------------------------------------------------------------

        if val_auc > best_val_auc:

            best_val_auc = val_auc

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        # ------------------------------------------------------------
        # EARLY STOPPING
        # ------------------------------------------------------------

        if patience_counter >= PATIENCE:

            print()
            print(
                "Early stopping triggered."
            )

            print(
                "Epoch:",
                epoch
            )

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    print()
    print(
        "Best Validation ROC AUC:",
        f"{best_val_auc:.6f}"
    )

    return model


# ====================================================================
# TEST
# ====================================================================

def evaluate_test(
    model,
    test_loader
):

    print()
    print("=" * 70)
    print("DCN TEST RESULTS")
    print("=" * 70)

    (
        test_loss,
        y_true,
        probabilities,
        predictions
    ) = evaluate(
        model,
        test_loader
    )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    print()
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

    print()
    print("Classification Report")

    print(
        classification_report(
            y_true,
            predictions,
            digits=6,
            zero_division=0
        )
    )

    print("Confusion Matrix")

    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )

    return (
        y_true,
        probabilities,
        predictions
    )


# ====================================================================
# SAVE PREDICTIONS
# ====================================================================

def save_predictions(
    y_true,
    probabilities,
    predictions
):

    prediction_df = pd.DataFrame({

        "actual":
            y_true,

        "dcn_probability":
            probabilities,

        "dcn_prediction":
            predictions
    })

    prediction_df.to_csv(
        PREDICTION_PATH,
        index=False
    )

    print()
    print(
        "Predictions saved to:"
    )

    print(
        PREDICTION_PATH
    )


# ====================================================================
# SAVE MODEL
# ====================================================================

def save_model(
    model,
    num_features
):

    os.makedirs(
        MODEL_SAVE_DIR,
        exist_ok=True
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "num_features":
            num_features,

        "num_cross_layers":
            NUM_CROSS_LAYERS,

        "deep_hidden_dims":
            DEEP_HIDDEN_DIMS,

        "dropout":
            DROPOUT,

        "model_type":
            "Deep Cross Network",

        "torch_version":
            torch.__version__,

        "cuda_version":
            torch.version.cuda,

        "device":
            str(DEVICE)
    }

    torch.save(
        checkpoint,
        MODEL_FILE
    )

    print()
    print(
        "Model saved to:"
    )

    print(
        MODEL_FILE
    )


# ====================================================================
# MAIN
# ====================================================================

def main():

    print()
    print("=" * 70)
    print("DCN FRAUD DETECTION")
    print("PHASE 3A")
    print("=" * 70)

    set_seed()

    # ---------------------------------------------------------------
    # DATA
    # ---------------------------------------------------------------

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    ) = load_data()

    num_features = X_train.shape[1]

    # ---------------------------------------------------------------
    # LOADERS
    # ---------------------------------------------------------------

    (
        train_loader,
        valid_loader,
        test_loader
    ) = create_dataloaders(
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    )

    # ---------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------

    model = create_model(
        num_features
    )

    # ---------------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------------

    model = train_model(
        model,
        train_loader,
        valid_loader
    )

    # ---------------------------------------------------------------
    # TEST
    # ---------------------------------------------------------------

    (
        y_true,
        probabilities,
        predictions
    ) = evaluate_test(
        model,
        test_loader
    )

    # ---------------------------------------------------------------
    # SAVE PREDICTIONS
    # ---------------------------------------------------------------

    save_predictions(
        y_true,
        probabilities,
        predictions
    )

    # ---------------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------------

    save_model(
        model,
        num_features
    )

    # ---------------------------------------------------------------
    # GPU MEMORY
    # ---------------------------------------------------------------

    if torch.cuda.is_available():

        print()

        print(
            "GPU peak memory allocated:",
            round(
                torch.cuda.max_memory_allocated()
                / (1024 ** 2),
                2
            ),
            "MB"
        )

    print()
    print("=" * 70)
    print("DCN PHASE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()