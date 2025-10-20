import pandas as pd

def test_data_shape():
    df = pd.read_csv("data/iris_1.csv")
    assert df.shape[1] == 5, "Unexpected number of columns"

def test_no_nulls():
    df = pd.read_csv("data/iris_1.csv")
    assert df.isnull().sum().sum() == 0, "Data contains null values"
