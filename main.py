from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

app = FastAPI()

# Input schema based on Data_Dictionary.csv
class LoanInput(BaseModel):
    Client_Income: float
    Car_Owned: int
    Bike_Owned: int
    Active_Loan: int
    House_Own: int
    Child_Count: int
    Credit_Amount: float
    Loan_Annuity: float
    Accompany_Client: str
    Client_Income_Type: str
    Client_Education: str
    Client_Marital_Status: str
    Client_Gender: str
    Loan_Contract_Type: str
    Client_Housing_Type: str
    Population_Region_Relative: float
    Age_Days: float
    Employed_Days: float
    Registration_Days: float
    ID_Days: float
    Own_House_Age: float
    Mobile_Tag: int
    Homephone_Tag: int
    Workphone_Working: int
    Client_Occupation: str
    Client_Family_Members: float
    Cleint_City_Rating: int
    Application_Process_Day: int
    Application_Process_Hour: int
    Client_Permanent_Match_Tag: str
    Client_Contact_Work_Tag: str
    Type_Organization: str
    Score_Source_1: float
    Score_Source_2: float
    Score_Source_3: float
    Social_Circle_Default: float
    Phone_Change: float
    Credit_Bureau: float

# Load your trained model (update path as needed)
# model = joblib.load('model.pkl')

@app.get("/")
def read_root():
    return {"message": "Loan Default Prediction API is running."}

@app.post("/predict")
def predict_default(data: LoanInput):
    # Convert input to DataFrame for model
    input_df = pd.DataFrame([data.dict()])
    # prediction = model.predict(input_df)[0]
    # probability = model.predict_proba(input_df)[0, 1]
    # For demonstration, return dummy values
    prediction = 0  # Replace with model prediction
    probability = 0.1  # Replace with model probability
    return {"default_prediction": int(prediction), "probability": float(probability)}

# Optionally, add a health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}
