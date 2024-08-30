# Title: Linear Regression from scratch with multiple independent variables
# Author: Emanuel Alvarez
# Origin Date: August 29, 2024

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load dataset in same directory as linear-regression.py
file_path = 'cleaned_carDetails.csv'
data = pd.read_csv(file_path)

# print(data.info()) # Show labels (columns) and their datatypes

# Determine the independent and dependent variables, seen in data.info() for linear regression
x = data[['Max Power', 'Year', 'Length']].values # Three features
y = data['Price'].values # Price as dependent variable y - numpy array with shape (n_samples)

# Normalize x data
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x = (x - x_mean) / x_std

# Normaliza y data
y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
y = (y - y_mean) / y_std

# Initialize parameters
w = np.zeros(x.shape[1]) # One weight for each feature
b = 0.0 # Constant added to weigthed feature value 
learning_rate = 0.01 # How much w and b should change (jump/step) with iteration. 0.001 to 1.0 but up to discretion.
epochs = 5000 # An epoch represents one complete pass through the entire training dataset during the training process.
             # During each epoch, the model’s parameters are updated based on the gradients computed from the training 
             # data. Multiple epochs are necessary to iteratively adjust the parameters to minimize the cost function.

# ////////////////////////////////////////////////////////////////////////////////
# computer_cost(x, y, w, b)
# 
# Purpose:
#       Calculate the Mean Squared Error (MSE) cost function for linear regression.
#       This measures how well the model's predictions match the actual target values.
# Algorithm: 
#       Mean Squared Error (MSE)
# Inputs:
#       x: Array of feature values (e.g., "Max Power" in your dataset).
#       y: Array of actual target values (e.g., "Price" in your dataset).
#       w: Weight parameter of the linear regression model.
#       b: Bias parameter of the linear regression model.
# Returns:
#       Total cost (MSE) calculated as the average of the squared differences between
#       the predicted values and the actual target values.
# ////////////////////////////////////////////////////////////////////////////////
def compute_cost(x, y, w, b):

    n = len(x) # number of data points

    prediction = np.dot(x, w) + b # Predictions now made based on many x's and w's

    squared_error = (prediction - y) ** 2 # Estimated value minus real value, squared

    total_cost = np.sum(squared_error) / (2 * n) # Compute MSE

    return total_cost
# end compute_cost()

# ////////////////////////////////////////////////////////////////////////////////
# Purpose:
#       Perform gradient descent to optimize the parameters w and b.
# Algorithm: 
#       Gradient Descent
# Inputs:
#       x: Array of feature values (input).
#       y: Array of actual target values (output).
#       w: Initial weight parameter of the linear regression model.
#       b: Initial bias parameter of the linear regression model.
#       learning_rate: The learning rate determines the step size during optimization.
#       epochs: Number of iterations to perform gradient descent.
# Outputs:
#       Printed statement declaring the epoch and current cost
# Returns:
#       tuple: The optimized weight (w) and bias (b).
# ////////////////////////////////////////////////////////////////////////////////
def gradient_descent(x, y, w, b, learning_rate, epochs):

    n = len(x) # number of data points

    # Train through number of epochs (iterations)
    for epoch in range(epochs):

        prediction = np.dot(x, w) + b # Predictions now made based on many x's and w's

        # Compute the gradient (partial derivatives) for w and b
        dw = np.dot(x.T, prediction - y) / n
        db = np.sum(prediction - y) / n
        
        # Update the parameters using the gradients and learning rate
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Print the cost every 100 epochs for monitoring
        if epoch % 500 == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Epoch {epoch}: Cost = {cost:.4f}")
    
    return w, b
# end gradient_descent()

# Save the model parameters to a file
def save_model(w, b, filename, x_mean, x_std, y_mean, y_std):
    with open(filename, 'wb') as file:
        pickle.dump({'w': w, 'b': b, 'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}, file)
    print(f"Model saved to {filename}")
# end save_model()

# Train the model
w, b = gradient_descent(x, y, w, b, learning_rate, epochs)

# Print Bias and list of weights
print("Trained parameters:")
print(f"Bias (b) = {b:.2f}")

for i, weight in enumerate(w):
    print(f"Weight {i+1} = {weight:.2f}")

# Save model
save_model(w, b, 'linear_regression__multiple_features_model.pkl', x_mean, x_std, y_mean, y_std)