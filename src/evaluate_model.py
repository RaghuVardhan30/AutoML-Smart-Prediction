from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METRICS_PATH = (
    PROJECT_ROOT
    / "models"
    / "evaluation_metrics.json"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "models"
    / "test_predictions.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "models"
)


# =========================================================
# LOAD RESULTS
# =========================================================

def load_results():

    print("Loading evaluation results...")

    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metrics = json.load(file)

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    return metrics, predictions


# =========================================================
# DISPLAY METRICS
# =========================================================

def display_metrics(metrics):

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"Model        : {metrics['model']}")
    print(f"Test Samples : {metrics['test_samples']}")
    print(f"MAE          : {metrics['mae']:.4f}")
    print(f"MSE          : {metrics['mse']:.4f}")
    print(f"RMSE         : {metrics['rmse']:.4f}")
    print(f"R²           : {metrics['r2']:.4f}")

    print("=" * 60)


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

def plot_actual_vs_predicted(predictions):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        predictions["Actual_G3"],
        predictions["Predicted_G3"]
    )

    # Perfect prediction line
    minimum = min(
        predictions["Actual_G3"].min(),
        predictions["Predicted_G3"].min()
    )

    maximum = max(
        predictions["Actual_G3"].max(),
        predictions["Predicted_G3"].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual Final Grade (G3)")
    plt.ylabel("Predicted Final Grade (G3)")
    plt.title("Actual vs Predicted Student Performance")

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "actual_vs_predicted.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.show()

    print(
        f"\nActual vs Predicted graph saved to:\n{output_path}"
    )


# =========================================================
# ERROR DISTRIBUTION
# =========================================================

def plot_error_distribution(predictions):

    plt.figure(figsize=(8, 6))

    plt.hist(
        predictions["Error"],
        bins=20
    )

    plt.xlabel("Prediction Error")
    plt.ylabel("Number of Students")
    plt.title("Prediction Error Distribution")

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "error_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.show()

    print(
        f"\nError distribution graph saved to:\n{output_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("SMART PREDICTION MODEL EVALUATION")
    print("=" * 60)

    metrics, predictions = load_results()

    display_metrics(metrics)

    print("\nFirst 10 predictions:")

    print(
        predictions
        .head(10)
        .to_string(index=False)
    )

    plot_actual_vs_predicted(
        predictions
    )

    plot_error_distribution(
        predictions
    )

    print("\nEvaluation completed successfully.")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()