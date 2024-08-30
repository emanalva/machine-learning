# Title: Linear Regression with machine learning python library: scikit-learn
# Author: Emanuel Alvarez
# Origin Date: August 30, 2024

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load your data from a CSV file into a DataFrame
data = pd.read_csv('cleaned_carDetails.csv')

# Define the features (independent variables) and the target (dependent variable)
x = data[['Max Power', 'Year', 'Length', 'Width', 'Height']]
y = data['Price']

# Initialize the StandardScaler for normalization
# scikit functions
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# Normalization: Fit to data and then transform
# scikit function
x_normalized = scaler_x.fit_transform(x)  # Normalize features
y_normalized = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()  # Normalize target

# Initialize and train the Linear Regression model
# scikit function
model = LinearRegression()
model.fit(x_normalized, y_normalized)

# Extract the model parameters
# scikit function
weights = model.coef_  # Coefficients of the features
bias = model.intercept_  # Bias term

# Save the model parameters and scalers to a pickle file
with open('linear_regression_lib_model.pkl', 'wb') as file:
    pickle.dump({
        'weights': weights,
        'bias': bias,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y
    }, file)

print("Model trained and parameters saved.")
print(f"Weights: {weights} and Bias: {bias}")