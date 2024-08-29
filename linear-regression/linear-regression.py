# Title: Linear Regression from scratch
# Author: Emanuel Alvarez
# Origin Date: August 28, 2024

# ////////////////////////////////////////////////////////////////////////////////
# Terminology:
# 
# Independent Variable (x): 
#       The variable used to predict the dependent variable.
# Dependent Variable (y): 
#       The variable we want to predict.
# Coefficient (w): 
#       Parameters of the model that we estimate from the data.
# Intercept (b): 
#       The value of y when x is 0.
# Cost Function: 
#       A measure of how well our model fits the data. For linear regression, we use Mean Squared Error (MSE).
# Gradient Descent: 
#       An optimization algorithm used to minimize the cost function by adjusting the model parameters.
# ////////////////////////////////////////////////////////////////////////////////

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load dataset in same directory as linear-regression.py
# Use the dataset to train the linear regression model
# This trained model can then be applied to new cases where some data is missing or unknown,
# allowing us to make predictions based on the relationships learned from the training data
file_path = 'cleaned_carDetails.csv'
data = pd.read_csv(file_path)

# print(data.head()) # Shows a small chunk of the intial entries in the dataset
# print(data.info()) # Show stats and information about the labels

# Determine the independent and dependent variables, seen in data.info() for linear regression
# Independent variables are the known or unchanged factors
# Dependent variables are what we aim to predict or understand from the data
x = data['Max Power'].values # Max power as independent variable x - numpy array with shape (n_samples, 1)
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
w = 0.0 # Weight multiplied to feature value to reach target value
b = 0.0 # Constant added to weigthed feature value 
learning_rate = 0.01 # How much w and b should change (jump/step) with iteration. 0.001 to 1.0 but up to discretion.
epochs = 1000 # An epoch represents one complete pass through the entire training dataset during the training process.
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

    prediction = x * w + b # Prediction = feature value, times the weight, plus the bias

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

        prediction = x * w + b # Prediction = feature value, times the weight, plus the bias

        # Compute the gradient (partial derivatives) for w and b
        dw = np.sum((prediction - y) * x) / n
        db = np.sum(prediction - y) / n
        
        # Update the parameters using the gradients and learning rate
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Print the cost every 100 epochs for monitoring
        if epoch % 100 == 0:
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
print(f"Trained parameters: w = {w:.2f}, b = {b:.2f}")
save_model(w, b, 'linear_regression_model.pkl', x_mean, x_std, y_mean, y_std)

# Plot the results
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, w * x + b, color='red', label='Regression Line')
plt.xlabel('Max Power')
plt.ylabel('Price')
plt.legend()
plt.title('Linear Regression on Car Data')
plt.show()