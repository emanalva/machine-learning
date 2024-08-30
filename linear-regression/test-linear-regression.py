# Title: Test script for linear regression models using .pkl file
# Author: Emanuel Alvarez
# Origin Date: August 29, 2024

import pickle

# Loads model parameters (weights, bias, standard deviations, and means) from a pickle file.
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model['w'], model['b'], model['x_mean'], model['x_std'], model['y_mean'], model['y_std']

# Predict the price based on the input power
def predict_price(max_power, w, b):
    return w * max_power + b

# Load the model into variables
w, b, x_mean, x_std, y_mean, y_std = load_model('linear_regression_model.pkl')

# Simple for-loop to display predicted prices given a range of maximum horepowers
print("////////////////////////////////////////////")
for hp in range (0, 300, 20):

    # Normalize input data using the independent variables mean and std from pickle file
    power_normalized = (hp - x_mean) / x_std

    # Predict normalized price
    price_normalized = predict_price(power_normalized, w, b)

    # Denormalize the price for display
    predicted_price = price_normalized * y_std + y_mean

    # Data set in INR. Convert to USD. 
    # As of writing script 1 INR = 0.012 USD
    predicted_price = predicted_price * 0.012

    # Display
    print(f"Predicted price ${predicted_price:>13,.2f} for bhp: {hp}")
print("////////////////////////////////////////////")
