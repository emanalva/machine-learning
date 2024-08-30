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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('titanic.csv')
print(data.head())
print(data.info())

# Preprocessing 

# Model training
