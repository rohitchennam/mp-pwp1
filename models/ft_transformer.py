"""
======================================================================
FT-TRANSFORMER FRAUD DETECTION
PHASE 3A

4-MODEL ARCHITECTURE:
    1. XGBoost
    2. MLP
    3. FT-Transformer
    4. DCN

CURRENT MODEL:
    FT-Transformer

DATASET:
    Training   : 259,070
    Validation : 37,010
    Testing    : 74,020
    Features   : 30 continuous features

IMPORTANT WINDOWS FIX:
    PyTorch MUST be imported before NumPy/Pandas/Scikit-learn.

PACKAGE:
    rtdl_revisiting_models==0.0.2
======================================================================
"""

# =====================================================================
# 1. WINDOWS / DLL ENVIRONMENT
# =====================================================================

import os
import sys
import site
import warnings

# These MUST be set before importing NumPy/Pandas/PyTorch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Prevent MKL/OpenMP thread conflicts
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

warnings.filterwarnings("ignore")


# =====================================================================
# 2. PROJECT ROOT
# =====================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =====================================================================
# 3. WINDOWS TORCH DLL PATH
#
# IMPORTANT:
# Add torch/lib to the DLL search path BEFORE importing torch.
# =====================================================================

def setup_torch_dll_path():

    if os.name != "nt":
        return

    possible_site_packages = []

    try:
        possible_site_packages.extend(
            site.getsitepackages()
        )
    except Exception:
        pass

    try:
        user_site = site.getusersitepackages()

        if user_site:
            possible_site_packages.append(
                user_site
            )
    except Exception:
        pass

    # Also check the Python executable location
    python_dir = os.path.dirname(
        sys.executable
    )

    possible_site_packages.append(
        os.path.join(
            python_dir,
            "Lib",
            "site-packages"
        )
    )

    for site_packages in possible_site_packages:

        torch_lib = os.path.join(
            site_packages,
            "torch",
            "lib"
        )

        if os.path.isdir(torch_lib):

            # Python 3.8+
            try:
                os.add_dll_directory(
                    torch_lib
                )
            except (AttributeError, OSError):
                pass

            # Also add to PATH
            current_path = os.environ.get(
                "PATH",
                ""
            )

            if torch_lib not in current_path:

                os.environ["PATH"] = (
                    torch_lib
                    + os.pathsep
                    + current_path
                )

            print(
                "Torch DLL directory:",
                torch_lib
            )

            return


setup_torch_dll_path()


# =====================================================================
# 4. CRITICAL:
# IMPORT PYTORCH BEFORE NUMPY / PANDAS / SKLEARN
# =====================================================================

try:

    import torch
    import torch.nn as nn

    from torch.utils.data import (
        TensorDataset,
        DataLoader
    )

except Exception as e:

    print()
    print("=" * 70)
    print("PYTORCH IMPORT FAILED")
    print("=" * 70)
    print()
    print("Python executable:")
    print(sys.executable)
    print()
    print("PyTorch import error:")
    print(repr(e))
    print()

    raise


# =====================================================================
# 5. NOW IMPORT NUMPY / PANDAS / SKLEARN
# =====================================================================

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
    confusion_matrix,
)


# =====================================================================
# 6. RTDL FT-TRANSFORMER
# =====================================================================

from rtdl_revisiting_models import FTTransformer


# =====================================================================
# 7. CONFIG
# =====================================================================

try:

    from config import (
        PROCESSED_DATA,
        MODEL_PATH
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


# =====================================================================
# 8. DATA PATHS
# =====================================================================

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


# =====================================================================
# 9. OUTPUT PATHS
# =====================================================================

PREDICTION_PATH = os.path.join(
    PROCESSED_DATA,
    "ft_transformer_test_predictions.csv"
)

MODEL_SAVE_DIR = MODEL_PATH

MODEL_FILE = os.path.join(
    MODEL_SAVE_DIR,
    "ft_transformer_model.pt"
)


# =====================================================================
# 10. FT-TRANSFORMER CONFIGURATION
# =====================================================================

D_BLOCK = 192

N_BLOCKS = 3

N_HEADS = 8

ATTENTION_DROPOUT = 0.10

FFN_DROPOUT = 0.10

RESIDUAL_DROPOUT = 0.00

# Official RTDL configuration supports this parameter
FFN_D_HIDDEN = None

FFN_D_HIDDEN_MULTIPLIER = 4 / 3

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.00001

BATCH_SIZE = 1024

EPOCHS = 15

PATIENCE = 4

RANDOM_STATE = 42


# =====================================================================
# 11. DEVICE
# =====================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================================
# 12. RANDOM SEED
# =====================================================================

def set_seed(
    seed=RANDOM_STATE
):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# =====================================================================
# 13. ENVIRONMENT TEST
# =====================================================================

def print_environment():

    print()
    print("=" * 70)
    print("FT-TRANSFORMER ENVIRONMENT")
    print("=" * 70)

    print(
        "Python executable :",
        sys.executable
    )

    print(
        "Python version    :",
        sys.version.split()[0]
    )

    print(
        "PyTorch version   :",
        torch.__version__
    )

    print(
        "CUDA available    :",
        torch.cuda.is_available()
    )

    print(
        "Device            :",
        DEVICE
    )

    print(
        "NumPy version     :",
        np.__version__
    )

    print(
        "Pandas version    :",
        pd.__version__
    )

    print("=" * 70)


# =====================================================================
# 14. LOAD DATA
# =====================================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING DATA FOR FT-TRANSFORMER")
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
    # TARGET
    # ---------------------------------------------------------------

    if isinstance(
        y_train,
        pd.DataFrame
    ):

        y_train = y_train.iloc[:, 0]

    if isinstance(
        y_valid,
        pd.DataFrame
    ):

        y_valid = y_valid.iloc[:, 0]

    if isinstance(
        y_test,
        pd.DataFrame
    ):

        y_test = y_test.iloc[:, 0]

    # ---------------------------------------------------------------
    # NUMERIC CONVERSION
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
    # INFINITY
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
    # TRAINING MEDIAN IMPUTATION
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
    # TARGET CONVERSION
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

    # ---------------------------------------------------------------
    # DATA VALIDATION
    # ---------------------------------------------------------------

    if X_train.shape[1] != X_valid.shape[1]:

        raise ValueError(
            "Training and validation feature counts differ."
        )

    if X_train.shape[1] != X_test.shape[1]:

        raise ValueError(
            "Training and testing feature counts differ."
        )

    # Ensure identical column order
    if list(X_train.columns) != list(X_valid.columns):

        raise ValueError(
            "Training and validation feature columns differ."
        )

    if list(X_train.columns) != list(X_test.columns):

        raise ValueError(
            "Training and testing feature columns differ."
        )

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
        y_test
    )


# =====================================================================
# 15. FT-TRANSFORMER CLASSIFIER
# =====================================================================

class FTTransformerClassifier(
    nn.Module
):

    def __init__(
        self,
        num_features
    ):

        super().__init__()

        self.num_features = (
            num_features
        )

        # -----------------------------------------------------------
        # IMPORTANT:
        # This is the API expected by
        # rtdl_revisiting_models==0.0.2
        # -----------------------------------------------------------

        self.transformer = FTTransformer(

            n_cont_features=num_features,

            cat_cardinalities=[],

            d_out=1,

            d_block=D_BLOCK,

            n_blocks=N_BLOCKS,

            attention_n_heads=N_HEADS,

            attention_dropout=ATTENTION_DROPOUT,

            ffn_d_hidden=FFN_D_HIDDEN,

            ffn_d_hidden_multiplier=(
                FFN_D_HIDDEN_MULTIPLIER
            ),

            ffn_dropout=FFN_DROPOUT,

            residual_dropout=RESIDUAL_DROPOUT
        )

    def forward(
        self,
        x
    ):

        output = self.transformer(
            x_cont=x,
            x_cat=None
        )

        return output.squeeze(-1)


# =====================================================================
# 16. CREATE MODEL
# =====================================================================

def create_model(
    num_features
):

    print()
    print("=" * 70)
    print("INITIALIZING FT-TRANSFORMER")
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

    print(
        "Input features  :",
        num_features
    )

    print(
        "D_BLOCK         :",
        D_BLOCK
    )

    print(
        "N_BLOCKS        :",
        N_BLOCKS
    )

    print(
        "N_HEADS         :",
        N_HEADS
    )

    print(
        "Learning rate   :",
        LEARNING_RATE
    )

    model = FTTransformerClassifier(
        num_features=num_features
    )

    model = model.to(
        DEVICE
    )

    print()
    print(
        "FT-Transformer initialized successfully"
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
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


# =====================================================================
# 17. CREATE DATALOADERS
# =====================================================================

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
        X_train.to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32
    )

    X_valid_tensor = torch.tensor(
        X_valid.to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test.to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32
    )

    y_valid_tensor = torch.tensor(
        y_valid.to_numpy(),
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test.to_numpy(),
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
        pin_memory=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
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


# =====================================================================
# 18. EVALUATION
# =====================================================================

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

        for (
            X_batch,
            y_batch
        ) in loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            logits = model(
                X_batch
            )

            if criterion is not None:

                loss = criterion(
                    logits,
                    y_batch
                )

                batch_size = (
                    X_batch.size(0)
                )

                total_loss += (
                    loss.item()
                    * batch_size
                )

                total_samples += (
                    batch_size
                )

            probabilities = torch.sigmoid(
                logits
            )

            all_probabilities.extend(
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

            all_targets.extend(
                y_batch
                .detach()
                .cpu()
                .numpy()
            )

    probabilities = np.asarray(
        all_probabilities,
        dtype=np.float64
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


# =====================================================================
# 19. TRAINING
# =====================================================================

def train_model(
    model,
    train_loader,
    valid_loader
):

    print()
    print("=" * 70)
    print("STARTING FT-TRANSFORMER TRAINING")
    print("=" * 70)

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    # ---------------------------------------------------------------
    # Use RTDL's recommended parameter groups when available.
    # This prevents inappropriate weight decay on normalization/bias
    # parameters.
    # ---------------------------------------------------------------

    if hasattr(
        model.transformer,
        "make_parameter_groups"
    ):

        parameter_groups = (
            model.transformer
            .make_parameter_groups()
        )

        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

    else:

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

        for (
            X_batch,
            y_batch
        ) in train_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
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

            batch_size = (
                X_batch.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

        train_loss = (
            running_loss
            / total_samples
        )

        # -----------------------------------------------------------
        # VALIDATION
        # -----------------------------------------------------------

        (
            val_loss,
            y_true,
            val_probability,
            val_prediction
        ) = evaluate(
            model,
            valid_loader,
            criterion
        )

        val_auc = roc_auc_score(
            y_true,
            val_probability
        )

        val_f1 = f1_score(
            y_true,
            val_prediction,
            zero_division=0
        )

        val_precision = precision_score(
            y_true,
            val_prediction,
            zero_division=0
        )

        val_recall = recall_score(
            y_true,
            val_prediction,
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

        # -----------------------------------------------------------
        # BEST MODEL
        # -----------------------------------------------------------

        if val_auc > best_val_auc:

            best_val_auc = val_auc

            best_state = {
                key: value.detach()
                .cpu()
                .clone()

                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        # -----------------------------------------------------------
        # EARLY STOPPING
        # -----------------------------------------------------------

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

    # ---------------------------------------------------------------
    # RESTORE BEST
    # ---------------------------------------------------------------

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


# =====================================================================
# 20. FINAL TEST EVALUATION
# =====================================================================

def evaluate_test(
    model,
    test_loader
):

    print()
    print("=" * 70)
    print("FT-TRANSFORMER TEST RESULTS")
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

    print(
        "Classification Report"
    )

    print(
        classification_report(
            y_true,
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
            y_true,
            predictions
        )
    )

    return (
        y_true,
        probabilities,
        predictions
    )


# =====================================================================
# 21. SAVE PREDICTIONS
# =====================================================================

def save_predictions(
    y_true,
    probabilities,
    predictions
):

    prediction_df = pd.DataFrame({

        "actual":
            y_true,

        "ft_transformer_probability":
            probabilities,

        "ft_transformer_prediction":
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


# =====================================================================
# 22. SAVE MODEL
# =====================================================================

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

        "d_block":
            D_BLOCK,

        "n_blocks":
            N_BLOCKS,

        "n_heads":
            N_HEADS,

        "attention_dropout":
            ATTENTION_DROPOUT,

        "ffn_dropout":
            FFN_DROPOUT,

        "residual_dropout":
            RESIDUAL_DROPOUT,

        "ffn_d_hidden":
            FFN_D_HIDDEN,

        "ffn_d_hidden_multiplier":
            FFN_D_HIDDEN_MULTIPLIER,

        "model_type":
            "FT-Transformer",

        "package":
            "rtdl_revisiting_models",

        "torch_version":
            torch.__version__,

        "random_state":
            RANDOM_STATE
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


# =====================================================================
# 23. MAIN
# =====================================================================

def main():

    print()

    print("=" * 70)
    print(
        "FT-TRANSFORMER FRAUD DETECTION"
    )
    print(
        "PHASE 3A"
    )
    print("=" * 70)

    # ---------------------------------------------------------------
    # ENVIRONMENT
    # ---------------------------------------------------------------

    print_environment()

    # ---------------------------------------------------------------
    # SEED
    # ---------------------------------------------------------------

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

    num_features = (
        X_train.shape[1]
    )

    # ---------------------------------------------------------------
    # DATALOADERS
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

    print()

    print("=" * 70)
    print(
        "FT-TRANSFORMER PHASE COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()