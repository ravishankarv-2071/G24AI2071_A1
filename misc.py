# misc.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def load_data():
    """
    Load Boston housing dataset from the original CMU URL (as specified in the assignment).
    Returns a pandas DataFrame with features + target column 'MEDV'.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    # split into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def get_X_y(df, target_col='MEDV'):
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    return X, y

def split_data(X, y, test_size=0.2, random_state=None):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(X_train, X_test, scaler=None):
    """
    Standard scale numeric features. Returns scaled arrays and the scaler.
    If a scaler is provided it will be used (so function is generic).
    """
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

def train_model(model, X_train, y_train):
    """
    Generic training wrapper. Accepts any sklearn-like estimator.
    """
    model.fit(X_train, y_train)
    return model

def test_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

def repeated_evaluate(model_factory, X, y, n_repeats=5, test_size=0.2, random_seed=42, preprocess=True):
    """
    Perform repeated random train/test splits and return average MSE,
    standard deviation, and list of mse per repeat.

    model_factory: callable that returns a fresh estimator when called (so function is generic).
    X: array-like or DataFrame
    y: array-like
    """
    mses = []
    rng = np.random.RandomState(random_seed)
    for i in range(n_repeats):
        rs = int(rng.randint(0, 1_000_000))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs)
        if preprocess:
            X_train, X_test, _ = preprocess_data(X_train, X_test)
        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        mses.append(mse)
    return float(np.mean(mses)), float(np.std(mses)), mses



