# Title: Linear Regression from scratch
# Author: Emanuel Alvarez
# Origin Date: August 28, 2024
# Most Recent Update: August 28, 2024

# Terminology:

# Dependent Variable (y): The variable we want to predict.
# Independent Variable (x): The variable used to predict the dependent variable.
# Coefficients (w): Parameters of the model that we estimate from the data.
# Intercept (b): The value of y when x is 0.
# Cost Function: A measure of how well our model fits the data. For linear regression, we use Mean Squared Error (MSE).
# Gradient Descent: An optimization algorithm used to minimize the cost function by adjusting the model parameters.

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset in same directory as linear-regresion.py
file_path = 'carDetails.csv'
data = pd.read_csv(file_path)

print(data.head())
print(data.info())