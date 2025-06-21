# Loan_Status_Prediction
It is a machine learning project with fastAPI (POST) use
The main objective of this project is to develop a loan approval prediction system using classification algorithms and deploy the model using FastAPI as a RESTful API.
**Key Highlights:
 Performed data preprocessing including:
 -Label encoding for categorical values.
 -Handling missing values by:
  Replacing with mode for columns with significant missing values.
  Dropping rows with very few missing values.

 Trained and evaluated multiple classification models
 -Saved the best model (RandomForestClassifier) using pickle.

 Built an API using FastAPI to serve the model and make real-time predictions.

** Technologies Used
    Python 3
    Pandas, NumPy — Data processing
    Scikit-learn — Model training and evaluation
    FastAPI — API development
    Uvicorn — ASGI server to run FastAPI
    Pickle — Model serialization
