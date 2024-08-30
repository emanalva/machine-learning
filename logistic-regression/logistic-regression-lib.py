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
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('titanic.csv')

# # Display basic statistics
# print(data.describe())

# # Check for correlations
# print(data.corr())

# # Plotting correlations
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
# plt.show()  # Explicitly display the heatmap
# plt.clf()

# Convert "Sex" and "Survived" to categorical types for better plotting
data['Sex'] = data['Sex'].astype('category')
data['Survived'] = data['Survived'].astype('category')

# Histograms for feature distribution
data[['Age', 'Fare', 'Sex', 'Sibsp', 'Parch', 'Pclass', 'Embarked', 'Survived']].hist()
plt.show()  # Display histograms

# Boxplot for Fare vs. Survived
plt.figure(figsize=(8, 6))
sns.boxplot(x='Survived', y='Fare', data=data)
plt.title('Boxplot of Fare vs. Survival')
plt.show()

# Count plot for Sex vs. Survived
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title('Count of Survival by Sex')
plt.show()

# Preprocessing 
# Missing data already removed, titanic.csv should have only relevant data
# Normalization

# Model training
