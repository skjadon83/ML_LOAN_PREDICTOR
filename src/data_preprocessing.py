import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data(path):
    """Load dataset from a CSV file."""
    return pd.read_csv(path)

def handle_missing_values(df):
    """Impute missing values: median for numeric, mode for categorical."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Numeric: median imputation
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Categorical: mode imputation
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df

def encode_categorical(df):
    """One-hot encode categorical variables."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

def preprocess_data(path):
    """Full preprocessing pipeline."""
    df = load_data(path)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    return df

if __name__ == "__main__":
    # Example usage
    data_path = "../data/Dataset.csv"
    df = preprocess_data(data_path)
    print("Preprocessed data shape:", df.shape)
    print(df.head())