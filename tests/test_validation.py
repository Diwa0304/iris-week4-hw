import pandas as pd
import os

DATA_FILE_NAME = os.environ.get("DATA_FILE_CI", "iris_1.csv")
DATA_PATH = os.path.join("data", DATA_FILE_NAME)

def test_data_shape():
    df = pd.read_csv("data/iris_1.csv")
    assert df.shape[1] == 5, "Unexpected number of columns"

def test_no_nulls():
    df = pd.read_csv("data/iris_1.csv")
    assert df.isnull().sum().sum() == 0, "Data contains null values"
