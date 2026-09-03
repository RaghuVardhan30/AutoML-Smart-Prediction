import os
import pandas as pd
from pycaret.regression import load_model, predict_model


# =========================================================
# MODEL PATH
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = load_model(MODEL_PATH)

print("Transformation Pipeline and Model Successfully Loaded")


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_grade(student_data):

    # Convert dictionary to DataFrame
    input_data = pd.DataFrame([student_data])

    # =====================================================
    # CURRENT MODEL FEATURES
    # =====================================================

    required_features = [
        "school",
        "sex",
        "age",
        "address",
        "famsize",
        "Pstatus",
        "Medu",
        "Fedu",
        "Mjob",
        "Fjob",
        "reason",
        "guardian",
        "traveltime",
        "studytime",
        "failures",
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "famrel",
        "freetime",
        "goout",
        "health",
        "absences"
    ]

    # =====================================================
    # CHECK FEATURES
    # =====================================================

    missing_features = [
        feature
        for feature in required_features
        if feature not in input_data.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Keep only model features
    input_data = input_data[required_features]

    # =====================================================
    # PREDICT
    # =====================================================

    prediction = predict_model(
        model,
        data=input_data
    )

    # Get prediction value
    predicted_grade = float(
        prediction["prediction_label"].iloc[0]
    )

    return round(predicted_grade, 2)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    sample_student = {

        "school": "GP",
        "sex": "M",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",

        "Medu": 4,
        "Fedu": 4,

        "Mjob": "teacher",
        "Fjob": "health",

        "reason": "course",
        "guardian": "mother",

        "traveltime": 1,
        "studytime": 3,
        "failures": 0,

        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",

        "famrel": 4,
        "freetime": 3,
        "goout": 3,

        "health": 4,
        "absences": 2
    }

    result = predict_grade(sample_student)

    print("=" * 50)
    print("AUTOML STUDENT PREDICTION")
    print("=" * 50)

    print(f"Predicted Grade: {result} / 20")