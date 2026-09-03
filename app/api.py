from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict_grade


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="AutoML Smart Student Prediction API",
    description="API for predicting a student's final academic grade using an AutoML model",
    version="1.0.0"
)


# =========================================================
# STUDENT INPUT MODEL
# =========================================================

class StudentData(BaseModel):

    school: str
    sex: str
    age: int
    address: str
    famsize: str
    Pstatus: str

    Medu: int
    Fedu: int

    Mjob: str
    Fjob: str

    reason: str
    guardian: str

    traveltime: int
    studytime: int
    failures: int

    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str

    famrel: int
    freetime: int
    goout: int
    health: int
    absences: int


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Welcome to the AutoML Smart Student Prediction API",
        "status": "API is running successfully"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "AutoML Student Performance Prediction Model"
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
def predict_student_grade(student: StudentData):

    try:

        # Convert Pydantic object into dictionary
        student_data = student.model_dump()

        # Predict grade
        predicted_grade = predict_grade(student_data)

        # Keep prediction within 0–20
        predicted_grade = max(0, min(20, predicted_grade))

        # Performance category
        if predicted_grade >= 16:
            category = "Excellent"

        elif predicted_grade >= 12:
            category = "Good"

        elif predicted_grade >= 8:
            category = "Average"

        else:
            category = "Needs Improvement"

        return {
            "predicted_final_grade": round(predicted_grade, 2),
            "maximum_grade": 20,
            "performance_category": category
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )