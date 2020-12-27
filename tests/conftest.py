import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing, fetch_covtype


def load_regression_data():
    dataset = fetch_california_housing(data_home="data", as_frame=True)
    df = dataset.frame.sample(5000)
    df["HouseAgeBin"] = pd.qcut(df["HouseAge"], q=4)
    df["HouseAgeBin"] = "age_" + df.HouseAgeBin.cat.codes.astype(str)
    test_idx = df.sample(int(0.2 * len(df)), random_state=42).index
    test = df[df.index.isin(test_idx)]
    train = df[~df.index.isin(test_idx)]
    return (train, test, dataset.target_names)


def load_classification_data():
    dataset = fetch_covtype(data_home="data")
    data = np.hstack([dataset.data, dataset.target.reshape(-1, 1)])[:10000, :]
    col_names = [f"feature_{i}" for i in range(data.shape[-1])]
    col_names[-1] = "target"
    data = pd.DataFrame(data, columns=col_names)
    data["feature_0_cat"] = pd.qcut(data["feature_0"], q=4)
    data["feature_0_cat"] = "feature_0_" + data.feature_0_cat.cat.codes.astype(str)
    test_idx = data.sample(int(0.2 * len(data)), random_state=42).index
    test = data[data.index.isin(test_idx)]
    train = data[~data.index.isin(test_idx)]
    return (train, test, ["target"])


@pytest.fixture(scope="session", autouse=True)
def regression_data():
    return load_regression_data()


@pytest.fixture(scope="session", autouse=True)
def classification_data():
    return load_classification_data()
