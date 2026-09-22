# AutoML Smart Prediction System with Deployment
1. Project Overview
The AutoML Smart Prediction System is a machine-learning application that predicts a student's final academic grade based on student-related academic, demographic, family, and social features.
The project uses PyCaret AutoML to compare different regression models and select a suitable model for predicting the student's final grade (G3).
The trained model is integrated with:
- Streamlit for the user interface
- FastAPI for the prediction API
- Docker for containerized deployment
- Git/GitHub for source-code version control
- DVC for dataset version control
- DagsHub for ML project and data management
2. Problem Statement
Predicting student academic performance can help identify students who may need additional academic support.
The objective of this project is to build an automated machine-learning system that:
1. Accepts student information as input.
2. Processes the input using the trained ML pipeline.
3. Predicts the student's final grade.
4. Displays the predicted grade out of 20.
5. Categorizes the predicted performance.
3. Objectives
The main objectives of this project are:
- To preprocess the student performance dataset.
- To select relevant features for prediction.
- To avoid target leakage.
- To use AutoML for regression model comparison.
- To select and tune a suitable machine-learning model.
- To evaluate the trained model.
- To create a prediction module.
- To develop a Streamlit user interface.
- To develop a FastAPI backend.
- To containerize the application using Docker.
- To version the dataset using DVC.
- To manage the ML project using GitHub and DagsHub.
4. Dataset
The project uses the UCI Student Performance dataset.
The original dataset contains:
Rows: 395
Columns: 33
The target variable is:
G3
G3 represents the student's final grade, with a maximum value of 20.
Original grade features
The dataset contains:
- G1 – first-period grade
- G2 – second-period grade
- G3 – final grade
G1 and G2 were removed from the final prediction features because they are previous-period grades that are strongly related to the final grade and could introduce target leakage.
5. Data Preprocessing
The preprocessing process is implemented in:
src/data_preprocessing.py
The preprocessing steps include:
1. Loading the dataset using Pandas.
2. Reading the semicolon-separated CSV file.
3. Checking the dataset shape.
4. Checking missing values.
5. Checking duplicate records.
6. Selecting G3 as the target variable.
7. Removing G1 and G2.
8. Removing romantic.
9. Removing Dalc and Walc.
10. Saving the processed dataset.
The processed dataset contains:
395 rows
28 columns
This consists of:
27 input features
+
1 target feature (G3)
The processed dataset is stored at:
data/processed/processed_data.csv
6. Features Used
The final input features are:
school
sex
age
address
famsize
Pstatus
Medu
Fedu
Mjob
Fjob
reason
guardian
traveltime
studytime
failures
schoolsup
famsup
paid
activities
nursery
higher
internet
famrel
freetime
goout
health
absences
The target variable is:
G3
7. AutoML
The project uses PyCaret Regression for the AutoML process.
The AutoML workflow includes:
Processed Dataset
       ↓
Train/Test Split
       ↓
Data Preprocessing
       ↓
Cross Validation
       ↓
Model Comparison
       ↓
Model Selection
       ↓
Hyperparameter Tuning
       ↓
Final Model
PyCaret automates several machine-learning tasks, including preprocessing, model comparison, evaluation, and tuning.
8. Model Selection
The AutoML process selected a Random Forest Regressor for the project.
The tuned model configuration used:
RandomForestRegressor(
    max_depth=5,
    min_impurity_decrease=0.01,
    min_samples_leaf=2,
    min_samples_split=7,
    n_estimators=280,
    n_jobs=-1,
    random_state=42
)
The final trained model is saved as:
models/best_model.pkl
9. Model Evaluation
The model is evaluated using regression metrics.
The main metrics are:
MAE
Mean Absolute Error measures the average absolute difference between the actual and predicted grades.
MSE
Mean Squared Error calculates the average squared prediction error.
RMSE
Root Mean Squared Error is the square root of MSE and represents the prediction error in the same unit as the grade.
R²
R² measures how much of the variation in the target variable is explained by the model.
The evaluation results are stored in:
models/evaluation_metrics.json
Additional evaluation outputs include:
models/test_predictions.csv
models/actual_vs_predicted.png
models/error_distribution.png
Model Performance

| Metric | Value |
|---|---:|
| MAE | 3.1961 |
| MSE | 15.2670 |
| RMSE | 3.9073 |
| R² | 0.2555 |
10. Prediction Module
The prediction functionality is implemented in:
src/predict.py
The module:
1. Loads the trained model.
2. Accepts student information.
3. Creates a Pandas DataFrame.
4. Checks the required features.
5. Passes the data through the trained PyCaret pipeline.
6. Generates the predicted final grade.
7. Returns the grade rounded to two decimal places.
Example:
Predicted Grade: 10.73 / 20
11. Streamlit Application
The user interface is implemented using:
app/streamlit_app.py
The Streamlit application allows users to enter student information through a graphical interface.
The application sends the student information to the prediction pipeline and displays:
Predicted Final Grade
Maximum Grade
Performance Category
The performance categories are:
Predicted Grade	Category
16–20	Excellent
12–15.99	Good
8–11.99	Average
Below 8	Needs Improvement


12. FastAPI Backend
The API is implemented in:
app/api.py
The FastAPI backend provides an endpoint for student-grade prediction.
Start FastAPI
uvicorn app.api:app --reload
The API runs at:
http://127.0.0.1:8000
Swagger Documentation
FastAPI automatically provides interactive API documentation at:
http://127.0.0.1:8000/docs
Prediction Endpoint
POST /predict
The endpoint accepts student information and returns a prediction.
Example response:
{
  "predicted_final_grade": 11.46,
  "maximum_grade": 20,
  "performance_category": "Average"
}
13. Testing
Unit testing is implemented in:
tests/test_prediction.py
The prediction functionality can be tested using:
pytest
The test verifies that the prediction module successfully loads the trained model and produces a prediction.
14. Docker
The application is containerized using Docker.
Docker files:
Dockerfile.api
Dockerfile.streamlit
docker-compose.yml
.dockerignore
The Docker Compose configuration runs:
FastAPI
   ↓
Port 8000

Streamlit
   ↓
Port 8501
Start Docker containers
docker compose up -d --build
Check containers
docker compose ps
Stop containers
docker compose down
Application URLs
Streamlit:
http://localhost:8501
FastAPI:
http://localhost:8000
FastAPI Swagger:
http://localhost:8000/docs
15. Git Version Control
Git is used to manage the source code of the project.
The project is maintained in GitHub and DagsHub.
The repository contains:
Source Code
Configuration Files
Docker Files
Model Files
DVC Metadata
Documentation
Tests
16. DVC Data Version Control
DVC is used to track the dataset instead of storing the dataset directly in Git.
The dataset is tracked using:
data/raw/dataset.csv.dvc
The actual dataset is stored in DVC remote storage.
The DVC remote is configured with DagsHub.
Check DVC status
dvc status
Push dataset to DVC remote
dvc push
This allows the dataset to be version controlled separately from the source code.
17. DagsHub
DagsHub is used for managing the machine-learning project and DVC data.
Project repository:
https://dagshub.com/RaghuVardhan30/AutoML-Smart-Prediction
The project uses:
Git → Source code
DVC → Dataset versioning
DagsHub → ML project/data management
The dataset itself is not stored directly inside Git.
Instead:
dataset.csv
     ↓
DVC
     ↓
DagsHub DVC Storage
while Git stores the DVC pointer:
dataset.csv.dvc
18. Project Structure
AutoML-Smart-Prediction/
│
├── app/
│   ├── __init__.py
│   ├── api.py
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   ├── dataset.csv
│   │   └── dataset.csv.dvc
│   │
│   └── processed/
│       └── processed_data.csv
│
├── models/
│   ├── best_model.pkl
│   ├── evaluation_metrics.json
│   ├── actual_vs_predicted.png
│   ├── error_distribution.png
│   └── test_predictions.csv
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_automl.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── tests/
│   └── test_prediction.py
│
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── requirements.txt
├── README.md
└── main.py
19. Installation
Clone the project repository and create a Python virtual environment.
Create virtual environment
python -m venv .venv
Activate virtual environment
.venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
20. Running the Project
Step 1: Preprocess the dataset
python src/data_preprocessing.py
Step 2: Train the AutoML model
python src/train_automl.py
Step 3: Evaluate the model
python src/evaluate_model.py
Step 4: Test prediction
python src/predict.py
Step 5: Run tests
pytest
Step 6: Start FastAPI
uvicorn app.api:app --reload
Step 7: Start Streamlit
In another terminal:
streamlit run app/streamlit_app.py
21. Complete System Workflow
UCI Student Dataset
        ↓
Data Preprocessing
        ↓
Feature Selection
        ↓
Processed Dataset
        ↓
PyCaret AutoML
        ↓
Model Comparison
        ↓
Random Forest Regressor
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Best Model
        ↓
Prediction Module
        ↓
 ┌───────────────┐
 │               │
 ▼               ▼
Streamlit      FastAPI
 │               │
 └───────┬───────┘
         ▼
       Docker
         │
         ▼
    Final Application
22. Technologies Used
Technology	Purpose
Python	Programming language
Pandas	Data processing
NumPy	Numerical operations
PyCaret	AutoML
Scikit-learn	Machine learning
Random Forest	Final regression model
Streamlit	User interface
FastAPI	REST API
Uvicorn	API server
Pytest	Testing
Docker	Containerization
Git	Source-code version control
GitHub	Code repository
DVC	Dataset version control
DagsHub	ML project and data management


23. Future Enhancements
Possible future improvements include:
- Improving model performance with additional feature engineering.
- Experimenting with additional regression algorithms.
- Adding model explainability.
- Adding more comprehensive automated testing.
- Deploying the Docker containers to a cloud platform.
- Adding monitoring for model performance.
- Adding new student-performance datasets.
24. Conclusion
The AutoML Smart Prediction System provides an end-to-end machine-learning workflow for predicting student final grades.
The project combines:
Data Preprocessing
        +
AutoML
        +
Model Evaluation
        +
Prediction
        +
Streamlit
        +
FastAPI
        +
Docker
        +
Git
        +
DVC
        +
DagsHub