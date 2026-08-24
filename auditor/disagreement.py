import os
import sys
import numpy as np
import pandas as pd


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
# PATHS
# ==========================================================

PROCESSED_DATA = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

XGB_FILE = os.path.join(
    PROCESSED_DATA,
    "xgb_test_predictions.csv"
)

MLP_FILE = os.path.join(
    PROCESSED_DATA,
    "mlp_test_predictions.csv"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_DATA,
    "xgb_mlp_disagreement.csv"
)


# ==========================================================
# LOAD PREDICTIONS
# ==========================================================

def load_predictions():

    print("=" * 70)
    print("LOADING MODEL PREDICTIONS")
    print("=" * 70)

    # ------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------

    print("\nLoading XGBoost predictions...")

    if not os.path.exists(XGB_FILE):

        raise FileNotFoundError(
            f"XGBoost prediction file not found:\n{XGB_FILE}"
        )

    xgb = pd.read_csv(
        XGB_FILE
    )

    print(
        f"XGBoost rows: {len(xgb):,}"
    )

    # ------------------------------------------------------
    # MLP
    # ------------------------------------------------------

    print("\nLoading MLP predictions...")

    if not os.path.exists(MLP_FILE):

        raise FileNotFoundError(
            f"MLP prediction file not found:\n{MLP_FILE}"
        )

    mlp = pd.read_csv(
        MLP_FILE
    )

    print(
        f"MLP rows: {len(mlp):,}"
    )

    # ------------------------------------------------------
    # Row count check
    # ------------------------------------------------------

    if len(xgb) != len(mlp):

        raise ValueError(
            "XGBoost and MLP prediction files "
            "contain different numbers of rows."
        )

    print(
        "\nBoth models contain the same "
        "number of test transactions."
    )

    return xgb, mlp


# ==========================================================
# CALCULATE DISAGREEMENT
# ==========================================================

def calculate_disagreement(
    xgb,
    mlp
):

    print("\n")
    print("=" * 70)
    print("CALCULATING MODEL DISAGREEMENT")
    print("=" * 70)

    # ======================================================
    # REQUIRED COLUMNS
    # ======================================================

    required_xgb = [
        "actual",
        "xgb_probability",
        "xgb_prediction"
    ]

    required_mlp = [
        "actual",
        "mlp_probability",
        "mlp_prediction"
    ]

    # ------------------------------------------------------
    # XGBoost columns
    # ------------------------------------------------------

    for column in required_xgb:

        if column not in xgb.columns:

            raise ValueError(
                f"Missing XGBoost column: {column}"
            )

    # ------------------------------------------------------
    # MLP columns
    # ------------------------------------------------------

    for column in required_mlp:

        if column not in mlp.columns:

            raise ValueError(
                f"Missing MLP column: {column}"
            )

    # ======================================================
    # VERIFY ACTUAL LABELS
    # ======================================================

    if not np.array_equal(
        xgb["actual"].values,
        mlp["actual"].values
    ):

        raise ValueError(
            "Actual labels do not match "
            "between XGBoost and MLP files."
        )

    # ======================================================
    # CREATE RESULT DATAFRAME
    # ======================================================

    result = pd.DataFrame()

    result["actual"] = (
        xgb["actual"].values
    )

    result["xgb_probability"] = (
        xgb["xgb_probability"].values
    )

    result["mlp_probability"] = (
        mlp["mlp_probability"].values
    )

    result["xgb_prediction"] = (
        xgb["xgb_prediction"].values
    )

    result["mlp_prediction"] = (
        mlp["mlp_prediction"].values
    )

    # ======================================================
    # PROBABILITY DIFFERENCE
    # ======================================================
    #
    # D = |P(XGB) - P(MLP)|
    #
    # Range:
    #
    # 0 → complete probability agreement
    # 1 → maximum probability disagreement
    #
    # ======================================================

    result["probability_difference"] = (
        np.abs(
            result["xgb_probability"]
            -
            result["mlp_probability"]
        )
    )

    # ======================================================
    # PREDICTION DISAGREEMENT
    # ======================================================

    result["prediction_disagreement"] = (
        result["xgb_prediction"]
        !=
        result["mlp_prediction"]
    ).astype(int)

    # ======================================================
    # DISAGREEMENT SCORE
    # ======================================================
    #
    # For the current two-model system:
    #
    # Disagreement Score =
    # |XGB Probability - MLP Probability|
    #
    # This is intentionally the ONLY score used
    # by our current auditor.
    #
    # ======================================================

    result["disagreement_score"] = (
        result["probability_difference"]
    )

    return result


# ==========================================================
# GENERAL DISAGREEMENT ANALYSIS
# ==========================================================

def analyze_disagreement(
    result
):

    print("\n")
    print("=" * 70)
    print("DISAGREEMENT ANALYSIS")
    print("=" * 70)

    total = len(result)

    # ======================================================
    # PREDICTION AGREEMENT
    # ======================================================

    prediction_disagreements = int(
        result[
            "prediction_disagreement"
        ].sum()
    )

    prediction_agreements = (
        total
        -
        prediction_disagreements
    )

    agreement_percentage = (
        prediction_agreements
        /
        total
        *
        100
    )

    disagreement_percentage = (
        prediction_disagreements
        /
        total
        *
        100
    )

    print(
        f"\nTotal transactions : {total:,}"
    )

    print(
        f"Prediction agreement    : "
        f"{agreement_percentage:.4f}%"
    )

    print(
        f"Prediction disagreement : "
        f"{disagreement_percentage:.4f}%"
    )

    # ======================================================
    # PROBABILITY DIFFERENCE STATISTICS
    # ======================================================

    mean_difference = (
        result[
            "disagreement_score"
        ].mean()
    )

    median_difference = (
        result[
            "disagreement_score"
        ].median()
    )

    std_difference = (
        result[
            "disagreement_score"
        ].std()
    )

    min_difference = (
        result[
            "disagreement_score"
        ].min()
    )

    max_difference = (
        result[
            "disagreement_score"
        ].max()
    )

    print(
        f"\nMean Disagreement Score   : "
        f"{mean_difference:.6f}"
    )

    print(
        f"Median Disagreement Score : "
        f"{median_difference:.6f}"
    )

    print(
        f"Std Disagreement Score    : "
        f"{std_difference:.6f}"
    )

    print(
        f"Minimum Disagreement Score: "
        f"{min_difference:.6f}"
    )

    print(
        f"Maximum Disagreement Score: "
        f"{max_difference:.6f}"
    )

    # ======================================================
    # TOP 20 OVERALL DISAGREEMENTS
    # ======================================================

    print("\n")
    print("=" * 70)
    print("TOP 20 DISAGREEMENTS")
    print("=" * 70)

    top_disagreements = (
        result
        .sort_values(
            "disagreement_score",
            ascending=False
        )
        .head(20)
    )

    print(
        top_disagreements[
            [
                "xgb_probability",
                "mlp_probability",
                "probability_difference",
                "xgb_prediction",
                "mlp_prediction",
                "actual",
                "disagreement_score"
            ]
        ].to_string(
            index=True
        )
    )

    # ======================================================
    # TOP 20 FRAUD DISAGREEMENTS
    # ======================================================

    print("\n")
    print("=" * 70)
    print("TOP 20 DISAGREEMENTS — FRAUD ANALYSIS")
    print("=" * 70)

    top_fraud = (
        result[
            result["actual"] == 1
        ]
        .sort_values(
            "disagreement_score",
            ascending=False
        )
        .head(20)
    )

    if len(top_fraud) == 0:

        print(
            "No fraud transactions found."
        )

    else:

        print(
            top_fraud[
                [
                    "xgb_probability",
                    "mlp_probability",
                    "probability_difference",
                    "xgb_prediction",
                    "mlp_prediction",
                    "actual",
                    "disagreement_score"
                ]
            ].to_string(
                index=True
            )
        )

    # ======================================================
    # TOP 20 LEGITIMATE DISAGREEMENTS
    # ======================================================

    print("\n")
    print("=" * 70)
    print("TOP 20 DISAGREEMENTS — LEGITIMATE ANALYSIS")
    print("=" * 70)

    top_legitimate = (
        result[
            result["actual"] == 0
        ]
        .sort_values(
            "disagreement_score",
            ascending=False
        )
        .head(20)
    )

    if len(top_legitimate) == 0:

        print(
            "No legitimate transactions found."
        )

    else:

        print(
            top_legitimate[
                [
                    "xgb_probability",
                    "mlp_probability",
                    "probability_difference",
                    "xgb_prediction",
                    "mlp_prediction",
                    "actual",
                    "disagreement_score"
                ]
            ].to_string(
                index=True
            )
        )


# ==========================================================
# THRESHOLD ANALYSIS
# ==========================================================

def threshold_analysis(
    result
):

    print("\n")
    print("=" * 70)
    print("DISAGREEMENT THRESHOLD ANALYSIS")
    print("=" * 70)

    # ======================================================
    # THRESHOLDS
    # ======================================================

    thresholds = [
        0.01,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
        0.70
    ]

    # ======================================================
    # TOTAL CLASS COUNTS
    # ======================================================

    total_fraud = int(
        (
            result["actual"] == 1
        ).sum()
    )

    total_legitimate = int(
        (
            result["actual"] == 0
        ).sum()
    )

    print(
        f"\nTotal fraud transactions      : "
        f"{total_fraud:,}"
    )

    print(
        f"Total legitimate transactions: "
        f"{total_legitimate:,}"
    )

    print("\n")

    # ======================================================
    # TABLE HEADER
    # ======================================================

    print(
        "Threshold | High Disagreement | "
        "Fraud | Legitimate | "
        "Fraud Capture % | "
        "Fraud % of High"
    )

    print("-" * 95)

    # ======================================================
    # ANALYZE EACH THRESHOLD
    # ======================================================

    for threshold in thresholds:

        high_disagreement = (
            result[
                "disagreement_score"
            ]
            >= threshold
        )

        # --------------------------------------------------
        # Fraud among high disagreement
        # --------------------------------------------------

        fraud_count = int(
            (
                high_disagreement
                &
                (
                    result["actual"] == 1
                )
            ).sum()
        )

        # --------------------------------------------------
        # Legitimate among high disagreement
        # --------------------------------------------------

        legitimate_count = int(
            (
                high_disagreement
                &
                (
                    result["actual"] == 0
                )
            ).sum()
        )

        # --------------------------------------------------
        # Total high disagreement
        # --------------------------------------------------

        total_high = int(
            high_disagreement.sum()
        )

        # --------------------------------------------------
        # Fraud capture rate
        # --------------------------------------------------

        if total_fraud > 0:

            fraud_capture = (
                fraud_count
                /
                total_fraud
                *
                100
            )

        else:

            fraud_capture = 0.0

        # --------------------------------------------------
        # Fraud percentage within high disagreement
        # --------------------------------------------------

        if total_high > 0:

            fraud_percentage = (
                fraud_count
                /
                total_high
                *
                100
            )

        else:

            fraud_percentage = 0.0

        print(
            f"{threshold:9.2f} | "
            f"{total_high:17,} | "
            f"{fraud_count:5,} | "
            f"{legitimate_count:10,} | "
            f"{fraud_capture:15.2f}% | "
            f"{fraud_percentage:13.2f}%"
        )


# ==========================================================
# SCORE RANGE ANALYSIS
# ==========================================================

def score_range_analysis(
    result
):

    print("\n")
    print("=" * 70)
    print("DISAGREEMENT SCORE RANGE ANALYSIS")
    print("=" * 70)

    ranges = [
        (0.00, 0.05),
        (0.05, 0.10),
        (0.10, 0.15),
        (0.15, 0.20),
        (0.20, 0.25),
        (0.25, 0.30),
        (0.30, 0.35),
        (0.35, 0.40),
        (0.40, 0.50),
        (0.50, 1.01)
    ]

    print(
        "Score Range | Total | Fraud | Legitimate | Fraud %"
    )

    print("-" * 65)

    for lower, upper in ranges:

        mask = (
            (result["disagreement_score"] >= lower)
            &
            (result["disagreement_score"] < upper)
        )

        total = int(
            mask.sum()
        )

        fraud = int(
            (
                mask
                &
                (result["actual"] == 1)
            ).sum()
        )

        legitimate = int(
            (
                mask
                &
                (result["actual"] == 0)
            ).sum()
        )

        if total > 0:

            fraud_percentage = (
                fraud
                /
                total
                *
                100
            )

        else:

            fraud_percentage = 0.0

        print(
            f"{lower:.2f} - {upper:.2f} | "
            f"{total:5,} | "
            f"{fraud:5,} | "
            f"{legitimate:10,} | "
            f"{fraud_percentage:7.2f}%"
        )


# ==========================================================
# SAVE RESULTS
# ==========================================================

def save_results(
    result
):

    os.makedirs(
        PROCESSED_DATA,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("DISAGREEMENT RESULTS SAVED")
    print("=" * 70)

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n")
    print("=" * 70)
    print("XGBOOST + MLP DISAGREEMENT ENGINE")
    print("=" * 70)

    # ======================================================
    # LOAD
    # ======================================================

    xgb, mlp = load_predictions()

    # ======================================================
    # CALCULATE DISAGREEMENT
    # ======================================================

    result = calculate_disagreement(
        xgb,
        mlp
    )

    # ======================================================
    # GENERAL ANALYSIS
    # ======================================================

    analyze_disagreement(
        result
    )

    # ======================================================
    # THRESHOLD ANALYSIS
    # ======================================================

    threshold_analysis(
        result
    )

    # ======================================================
    # SCORE RANGE ANALYSIS
    # ======================================================

    score_range_analysis(
        result
    )

    # ======================================================
    # SAVE
    # ======================================================

    save_results(
        result
    )

    # ======================================================
    # COMPLETE
    # ======================================================

    print("\n")
    print("=" * 70)
    print("DISAGREEMENT ENGINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()