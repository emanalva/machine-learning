# Title: Logistic Regression using Libraries
# Author: Emanuel Alvarez
# Origin Date: August 30, 2024

# ////////////////////////////////////////////////////////////////////////////////
# Legend for dataset:
#       - 'Passengerid': id unique to each passenger
#       - 'Age': passenger age
#       - 'Fare': passenger ticket price
#       - 'Sex': passenger sex (0: male or 1: female)
#       - 'Sibsp': num of siblings/spouses onboard (relative to passenger)
#       - 'Parch': num of parents/children onboard (relative to passenger)
#       - 'Pclass': socioeconomic class (1: low, 2: middle, or 3: high)
#       - 'Embarked': which port passenger embarked from (0: Cherbourg, 1: Queenstown, or 2: Southampton)
#       - 'Survived': passenger survival (0: no or 1: yes)
# ////////////////////////////////////////////////////////////////////////////////

# Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np


# Load the dataset
data = pd.read_csv('titanic.csv')

# ////////////////////////////////////////////////////////////////////////////////
# Commented out code to analyze data

# # Display basic statistics
# print(data.describe())

# # Check for correlations
# print(data.corr())

# # Plotting correlations
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
# plt.show()  # Explicitly display the heatmap
# plt.clf()

# # Histograms for feature distribution
# data[['Age', 'Fare', 'Sex', 'Sibsp', 'Parch', 'Pclass', 'Embarked', 'Survived']].hist()
# plt.show()  # Display histograms

# # Boxplot for Fare vs. Survived
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='Survived', y='Fare', data=data)
# plt.title('Boxplot of Fare vs. Survival')
# plt.show()

# # Count plot for Sex vs. Survived
# sns.countplot(x='Sex', hue='Survived', data=data)
# plt.title('Count of Survival by Sex')
# plt.show()
# ////////////////////////////////////////////////////////////////////////////////

# Preprocessing 
# Missing data already removed, titanic.csv should have only relevant data

# Choose features
# I'm going to run two models to see impact on accuracy:
#       1. without Fare
#       2. with Fare
x = data[['Sex']]
x_fare = data[['Sex', 'Fare']] 

# Choose target
y = data['Survived']

# Split data intro 80% training and 20% testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_fare_train, x_fare_test, y_fare_train, y_fare_test = train_test_split(x_fare, y, test_size=0.2, random_state=42)

# Initialize the logistic regression models
model_x = LogisticRegression(max_iter=1000, class_weight='balanced')
model_x_fare = LogisticRegression(max_iter=1000, class_weight='balanced')

# Train the models
model_x.fit(x_train, y_train)
model_x_fare.fit(x_fare_train, y_fare_train)

# Make predictions
predict_y_x = model_x.predict(x_test)
predict_y_x_fare = model_x_fare.predict(x_fare_test)

# Evaluate the models
print("Model without Fare")
print("Accuracy:", accuracy_score(y_test, predict_y_x))
print("Classification Report:\n", classification_report(y_test, predict_y_x))
print("Model with Fare")
print("Accuracy:", accuracy_score(y_fare_test, predict_y_x_fare))
print("Classification Report:\n", classification_report(y_fare_test, predict_y_x_fare))

# Plot confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, predict_y_x), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (without Fare)')

sns.heatmap(confusion_matrix(y_fare_test, predict_y_x_fare), annot=True, fmt='d', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (with Fare)')

plt.show()