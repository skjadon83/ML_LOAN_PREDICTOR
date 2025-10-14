import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

def load_preprocessed_data(path):
    """Load preprocessed data from CSV."""
    return pd.read_csv(path)

def train_model(df, target_col='Default'):
    """Train a RandomForest model with SMOTE for class imbalance."""
    X = df.drop(columns=[target_col, 'ID'], errors='ignore')
    y = df[target_col]

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_proba))

    return clf

if __name__ == "__main__":
    # Example usage
    data_path = "../data/Dataset.csv"
    # You may want to use the output of your preprocessing pipeline instead
    df = pd.read_csv(data_path)
    model = train_model(df)
    # Save the trained model
    joblib.dump(model, "../models/random_forest_model.joblib")