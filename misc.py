import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def load_data() -> pd.DataFrame:
    """Load the Boston Housing dataset from CMU (since sklearn's load_boston is deprecated)."""
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None, engine="python")

    # Reconstruct features (13 columns) and target (MEDV) from the CMU format
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE",
                     "DIS","RAD","TAX","PTRATIO","B","LSTAT"]
    df = pd.DataFrame(data, columns=feature_names)
    df["MEDV"] = target
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_pipeline(estimator, scale: bool = True) -> Pipeline:
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)

def train_and_eval(pipeline: Pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse, pipeline
