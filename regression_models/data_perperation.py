import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def prepare_data(file_path="data/predict_home_value.csv"):
    # load data
    df = pd.read_csv(file_path)

    # drop unnecessary columns
    df = df.drop(['ID'], axis=1)

    # Define categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = [col for col in df.select_dtypes(include=['float', 'int']).columns if col != 'SALEPRICE']

    # preprocessor for pipelines
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder="passthrough"
    )

    # split data
    X = df.drop('SALEPRICE', axis=1)
    y = df['SALEPRICE']
    return train_test_split(X, y, test_size=0.2, random_state=0), preprocessor
