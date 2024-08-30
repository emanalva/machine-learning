# Title: Test script for linear regression with multiple features models using .pkl file
# Author: Emanuel Alvarez
# Origin Date: August 29, 2024

import pickle
import numpy as np

# Loads model parameters (weights, bias, standard deviations, and means) from a pickle file.
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model['w'], model['b'], model['x_mean'], model['x_std'], model['y_mean'], model['y_std']

# Predict price for given features
def predict_price(features, w, b):
    return np.dot(features, w) + b

# Load the model into variables
w, b, x_mean, x_std, y_mean, y_std = load_model('linear_regression__multiple_features_model.pkl')

# 2D array where each row represents the features of a car
# Example: [Max Power (hp), Year, Length (mm), Width (mm), Height (mm)]
features_array = np.array([
    [87, 2017, 3990, 1680, 1505],  # Car 1
    [74, 2014, 3995, 1695, 1555],  # Car 2
    [79, 2011, 3585, 1595, 1550],  # Car 3
    [82, 2019, 3995, 1745, 1510],  # Car 4
    [148, 2018, 4735, 1830, 1795],  # Car 5
    [91, 2017, 4490, 1730, 1485],   # Car 6
    [205, 2007, 4404, 1877, 1800]  # Car 7 
])

# Normalize input data using the independent variables mean and std from pickle file
features_normalized = (features_array - x_mean) / x_std

# Predict normalized price for all cars
prices_normalized = predict_price(features_normalized, w, b)

# Denormalize the prices for display
predicted_prices = prices_normalized * y_std + y_mean

# Convert to USD from INR (at time of creation 1 INR = 0.012 USD)
predicted_prices = predicted_prices * 0.012

# Print predicted prices with corresponding features
for i, features in enumerate(features_array):
    print(f"Features {features} yield a predicted price of |${predicted_prices[i]:<10,.2f}|")