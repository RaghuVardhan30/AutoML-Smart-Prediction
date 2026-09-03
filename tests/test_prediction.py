import sys
import os

# Add project root directory to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from src.predict import predict_grade


def test_prediction():

    # Sample student input
    student_data = {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 4,
        "Fedu": 4,
        "Mjob": "teacher",
        "Fjob": "services",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 3,
        "failures": 0,
        "schoolsup": "yes",
        "famsup": "yes",
        "paid": "yes",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "famrel": 4,
        "freetime": 3,
        "goout": 2,
        "health": 5,
        "absences": 2
    }

    # Get prediction
    prediction = predict_grade(student_data)

    print("=" * 50)
    print("TESTING AUTOML PREDICTION SYSTEM")
    print("=" * 50)

    print(f"\nPredicted Grade: {prediction} / 20")

    # Test 1: Prediction must be a number
    assert isinstance(prediction, float)

    # Test 2: Prediction must be between 0 and 20
    assert 0 <= prediction <= 20

    print("\nAll tests passed successfully! ✅")


if __name__ == "__main__":
    test_prediction()