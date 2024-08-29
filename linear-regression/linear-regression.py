# Title: Linear Regression from scratch
# Author: Emanuel Alvarez
# Origin Date: August 28, 2024

# ////////////////////////////////////////////////////////////////////////////////
# Terminology:
# Dependent Variable (y): The variable we want to predict.
# Independent Variable (x): The variable used to predict the dependent variable.
# Coefficients (w): Parameters of the model that we estimate from the data.
# Intercept (b): The value of y when x is 0.
# Cost Function: A measure of how well our model fits the data. For linear regression, we use Mean Squared Error (MSE).
# Gradient Descent: An optimization algorithm used to minimize the cost function by adjusting the model parameters.
# ////////////////////////////////////////////////////////////////////////////////

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset in same directory as linear-regresion.py
# Use the dataset to train the linear regression model
# This trained model can then be applied to new cases where some data is missing or unknown,
# allowing us to make predictions based on the relationships learned from the training data
file_path = 'carDetails.csv'
data = pd.read_csv(file_path)

print(data.head()) # Shows a small chunk of the intial entries in the dataset
print(data.info()) # Show stats and information about the labels

# Determine the independent and dependent variables, seen in data.info() for linear regression
# Independent variables are the known or unchanged factors
# Dependent variables are what we aim to predict or understand from the data
X = data[['Max Power']].values  # Max power as independent variable x - numpy array with shape (n_samples, 1)
y = data['Price'].values        # Price as dependent variable y - numpy array with shape (n_samples)

# Initialize parameters
w = 0.0 # Weight multiplied to feature value to reach target value
b = 0.0 # Constant added to weigthed feature value 
learning_rate = 0.01 # How much w and b should change (jump/step) with iteration. 0.001 to 1.0 but up to discretion.
epoch = 1000 # An epoch represents one complete pass through the entire training dataset during the training process.
             # During each epoch, the model’s parameters are updated based on the gradients computed from the training 
             # data. Multiple epochs are necessary to iteratively adjust the parameters to minimize the cost function.