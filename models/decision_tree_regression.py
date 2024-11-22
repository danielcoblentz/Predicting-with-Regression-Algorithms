# Add necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load housing dataset
df = pd.read_csv("https://raw.githubusercontent.com/IBM/ml-learning-path-assets/master/data/predict_home_value.csv")
df = df.drop(['ID'], axis=1)

# apply a log transformation to the target variable (SALESPRICE) for normalization
df['LOG_SALEPRICE'] = np.log(df['SALEPRICE'])

#Prepare features (X) and target variable (y) for model training 
X = df.drop(['SALEPRICE', 'LOG_SALEPRICE'], axis=1)
y = df['LOG_SALEPRICE']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols), # standardize numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Decision Tree with constraints (add preprocessing)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42))
])

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
model.fit(X_train, y_train)

# Predict the LOG_SALEPRICE for testing data
y_pred = model.predict(X_test)

# output the performance metrics of the model
print("Decision Tree Regression")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
