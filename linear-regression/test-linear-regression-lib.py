# Title: Test Linear Regression Model with scikit-learn
# Author: Emanuel Alvarez
# Origin Date: August 30, 2024

import numpy as np
import pickle

# Load the model parameters and scalers from the pickle file
with open('linear_regression_lib_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

weights = model_data['weights']
bias = model_data['bias']
scaler_x = model_data['scaler_x']
scaler_y = model_data['scaler_y']

# Define new feature data to test the model
features_array = np.array([
    [87, 2017, 3990, 1680, 1505],  # Car 1
    [74, 2014, 3995, 1695, 1555],  # Car 2
    [79, 2011, 3585, 1595, 1550],  # Car 3
    [82, 2019, 3995, 1745, 1510],  # Car 4
    [148, 2018, 4735, 1830, 1795],  # Car 5
    [91, 2017, 4490, 1730, 1485],   # Car 6
    [205, 2007, 4404, 1877, 1800]  # Car 7 
])

# Normalize the feature data using the same scalers
features_normalized = scaler_x.transform(features_array)

# Predict prices using the trained model
predicted_prices_normalized = np.dot(features_normalized, weights) + bias

# Denormalize the predicted prices to get the original scale
predicted_prices = scaler_y.inverse_transform(predicted_prices_normalized.reshape(-1, 1)).flatten()

# Convert the prices from INR to USD (1 INR = 0.012 USD)
predicted_prices_usd = predicted_prices * 0.012

# Display the results
for i, features in enumerate(features_array):
    print(f"Features: {features} -> Predicted price: ${predicted_prices_usd[i]:,.2f}")
