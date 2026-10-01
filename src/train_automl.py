from pathlib import Path
import json

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from pycaret.regression import (
    setup,
    compare_models,
    tune_model,
    finalize_model,
    save_model,
    predict_model
)


# =========================================================
# MLFLOW CONFIGURATION
# =========================================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("AutoML Student Prediction")


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processed_data.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "models"
)

METRICS_PATH = (
    RESULTS_DIR
    / "evaluation_metrics.json"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "test_predictions.csv"
)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("AUTOML SMART PREDICTION SYSTEM")
    print("=" * 60)

    # -----------------------------------------------------
    # START MLFLOW RUN
    # -----------------------------------------------------

    with mlflow.start_run(run_name="AutoML Student Prediction"):

        print("\nMLflow run started.")

        # -------------------------------------------------
        # 1. Load processed dataset
        # -------------------------------------------------

        print("\n[1] Loading processed dataset...")

        df = pd.read_csv(DATA_PATH)

        print(f"Dataset shape: {df.shape}")
        print("\nTarget variable: G3")

        # Log dataset information
        mlflow.log_param("dataset_rows", df.shape[0])
        mlflow.log_param("dataset_columns", df.shape[1])
        mlflow.log_param("target_variable", "G3")

        # -------------------------------------------------
        # 2. Separate features and target
        # -------------------------------------------------

        X = df.drop(columns=["G3"])
        y = df["G3"]

        print(f"\nFeatures: {X.shape[1]}")
        print(f"Target samples: {len(y)}")

        mlflow.log_param("number_of_features", X.shape[1])
        mlflow.log_param("number_of_samples", len(y))

        # -------------------------------------------------
        # 3. Train-Test Split
        # -------------------------------------------------

        print("\n[2] Splitting dataset...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples : {len(X_test)}")

        # Log split parameters
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("testing_samples", len(X_test))

        # -------------------------------------------------
        # 4. Create training dataframe
        # -------------------------------------------------

        train_data = X_train.copy()

        train_data["G3"] = y_train.values

        # -------------------------------------------------
        # 5. AutoML Setup
        # -------------------------------------------------

        print("\n[3] Initializing AutoML...")

        categorical_features = [
            "school",
            "sex",
            "address",
            "famsize",
            "Pstatus",
            "Mjob",
            "Fjob",
            "reason",
            "guardian",
            "schoolsup",
            "famsup",
            "paid",
            "activities",
            "nursery",
            "higher",
            "internet",
        ]

        automl_setup = setup(
            data=train_data,
            target="G3",

            session_id=42,

            categorical_features=categorical_features,

            normalize=True,
            normalize_method="zscore",

            fold=10,

            verbose=True
        )

        # Log AutoML configuration
        mlflow.log_param("automl_framework", "PyCaret")
        mlflow.log_param("session_id", 42)
        mlflow.log_param("cross_validation_folds", 10)
        mlflow.log_param("normalization", True)
        mlflow.log_param("normalization_method", "zscore")

        # -------------------------------------------------
        # 6. Compare Models
        # -------------------------------------------------

        print("\n[4] Comparing machine learning models...")

        best_model = compare_models(
            sort="RMSE"
        )

        print("\n" + "=" * 60)
        print("BEST MODEL FROM AUTOML")
        print("=" * 60)

        print(best_model)

        # Get model name
        best_model_name = type(best_model).__name__

        print(f"\nSelected model: {best_model_name}")

        mlflow.log_param(
            "best_model",
            best_model_name
        )

        # -------------------------------------------------
        # 7. Tune Best Model
        # -------------------------------------------------

        print("\n[5] Tuning best model...")

        tuned_model = tune_model(
            best_model,
            optimize="RMSE"
        )

        print("\nTuned model:")
        print(tuned_model)

        tuned_model_name = type(tuned_model).__name__

        mlflow.log_param(
            "tuned_model",
            tuned_model_name
        )

        # -------------------------------------------------
        # 8. Predict on unseen test data
        # -------------------------------------------------

        print("\n[6] Evaluating model on unseen test data...")

        test_data = X_test.copy()

        test_data["G3"] = y_test.values

        predictions = predict_model(
            tuned_model,
            data=test_data
        )

        # PyCaret normally stores predictions
        # in "prediction_label"

        prediction_column = "prediction_label"

        if prediction_column not in predictions.columns:
            raise ValueError(
                "Prediction column was not generated by PyCaret."
            )

        y_pred = predictions[prediction_column].values

        # -------------------------------------------------
        # 9. Calculate evaluation metrics
        # -------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            y_pred
        )

        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)

        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R²   : {r2:.4f}")

        # -------------------------------------------------
        # 10. Log evaluation metrics to MLflow
        # -------------------------------------------------

        print("\nLogging metrics to MLflow...")

        mlflow.log_metric(
            "test_mae",
            float(mae)
        )

        mlflow.log_metric(
            "test_mse",
            float(mse)
        )

        mlflow.log_metric(
            "test_rmse",
            float(rmse)
        )

        mlflow.log_metric(
            "test_r2",
            float(r2)
        )

        # -------------------------------------------------
        # 11. Save evaluation metrics
        # -------------------------------------------------

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        metrics = {
            "model": str(tuned_model),
            "test_samples": int(len(y_test)),
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2)
        }

        with open(
            METRICS_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        print(
            f"\nEvaluation metrics saved to:\n{METRICS_PATH}"
        )

        # -------------------------------------------------
        # 12. Save test predictions
        # -------------------------------------------------

        prediction_results = pd.DataFrame({
            "Actual_G3": y_test.values,
            "Predicted_G3": y_pred
        })

        prediction_results["Error"] = (
            prediction_results["Actual_G3"]
            - prediction_results["Predicted_G3"]
        )

        prediction_results.to_csv(
            PREDICTIONS_PATH,
            index=False
        )

        print(
            f"Test predictions saved to:\n{PREDICTIONS_PATH}"
        )

        # -------------------------------------------------
        # 13. Finalize model
        # -------------------------------------------------

        print("\n[7] Finalizing model...")

        final_model = finalize_model(
            tuned_model
        )

        # -------------------------------------------------
        # 14. Save final model
        # -------------------------------------------------

        save_model(
            final_model,
            str(MODEL_PATH)
        )

        print(
            f"\nFinal model saved to:\n{MODEL_PATH}.pkl"
        )

        # -------------------------------------------------
        # 15. Log model to MLflow
        # -------------------------------------------------

        print("\nLogging model to MLflow...")

        mlflow.sklearn.log_model(
            final_model,
            name="model",
            serialization_format="cloudpickle",
            registered_model_name="AutoML-Student-Prediction"
        )

        # -------------------------------------------------
        # 16. Log output files as MLflow artifacts
        # -------------------------------------------------

        print("\nLogging artifacts to MLflow...")

        if METRICS_PATH.exists():
            mlflow.log_artifact(
                str(METRICS_PATH),
                artifact_path="metrics"
            )

        if PREDICTIONS_PATH.exists():
            mlflow.log_artifact(
                str(PREDICTIONS_PATH),
                artifact_path="predictions"
            )

        # -------------------------------------------------
        # 17. Display sample predictions
        # -------------------------------------------------

        print("\n" + "=" * 60)
        print("SAMPLE PREDICTIONS")
        print("=" * 60)

        print(
            prediction_results
            .head(10)
            .to_string(index=False)
        )

        # -------------------------------------------------
        # 18. MLflow run information
        # -------------------------------------------------

        run = mlflow.active_run()

        if run is not None:

            print("\n" + "=" * 60)
            print("MLFLOW INFORMATION")
            print("=" * 60)

            print(
                f"Run ID: {run.info.run_id}"
            )

            print(
                "Experiment: AutoML Student Prediction"
            )

            print(
                "MLflow UI: http://127.0.0.1:5000"
            )

        print("\n" + "=" * 60)
        print("AUTOML TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()