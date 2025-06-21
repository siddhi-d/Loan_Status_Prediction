from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load('loan_status_predictor.pkl')
app = FastAPI()

num_col = num_col = ["ApplicantIncome","CoapplicantIncome","LoanAmount" ,"Loan_Amount_Term"]
scaler = joblib.load('vector.pkl')

class LoanApproval(BaseModel):
    Gender: float
    Married: float
    Dependents: float
    Education: float
    Self_Employed : float
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term:float
    Credit_History: float
    Property_Area: float

@app.post("/predict")
async def predict_loan_status(application: LoanApproval):
   input_data =  pd.dataFrame([application.dict()])
   input_data[num_col] = scaler.transform(input_data[num_col])

   result = model.predict(input_data)

   if result[0] == 1:
       return {'Loan Status':"Approved"}
   else:
       return {'Loan Status':"Not Approved"}