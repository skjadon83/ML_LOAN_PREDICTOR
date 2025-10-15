# Loan Default Prediction Project

This project predicts the likelihood of a borrower defaulting on a loan using machine learning. It includes data preprocessing, feature engineering, model training, evaluation, and deployment via FastAPI.

## Project Structure

```
loan-default-prediction/
│
├── data/
│   ├── Dataset.csv
│   └── Data_Dictionary.csv
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── main.py                # FastAPI app
├── requirements.txt
├── Dockerfile
└── README.md
```

## Data
- **Dataset.csv**: Main dataset for training and evaluation.
- **Data_Dictionary.csv**: Describes each variable in the dataset.

## Setup Instructions

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run EDA**
   - Open `notebooks/exploratory_analysis.ipynb` in Jupyter or VS Code.
4. **Preprocess Data**
   ```bash
   python src/data_preprocessing.py
   ```
5. **Train Model**
   ```bash
   python src/model_training.py
   ```
6. **Evaluate Model**
   ```bash
   python src/model_evaluation.py
   ```
7. **Run API (FastAPI)**
   ```bash
   uvicorn main:app --reload
   ```
8. **Docker (optional)**
   ```bash
   docker build -t loan-default-api .
   docker run -p 8000:8000 loan-default-api
   ```

## API Endpoints
- `GET /` : Welcome message
- `POST /predict` : Predict loan default (send JSON with all required fields)
- `GET /health` : Health check

## Example Data Dictionary
See `data/Data_Dictionary.csv` for full details.

| Column Name                | Description                                      |
|---------------------------|--------------------------------------------------|
| Client_Income             | Client Income in $                               |
| Car_Owned                 | Car owned (0=No, 1=Yes)                          |
| ...                       | ...                                              |
| Default                   | Target: 1=Defaulted, 0=Not Defaulted             |

## Requirements
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
- fastapi
- uvicorn
- joblib
- imbalanced-learn

## Notes
- Update `main.py` to load your trained model for real predictions.
- See the notebook for EDA and feature engineering steps.
- The pipeline handles missing values, outliers, and class imbalance.

---

For more details, see the code and comments in each file.