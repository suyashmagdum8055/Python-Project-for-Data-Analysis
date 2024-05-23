# Python-Project-for-Data-Analysis
This Project Which creates for Python data analysis
# Step 1: Data Collection
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the Titanic dataset from Seaborn library
titanic = sns.load_dataset('titanic')

# Step 2: Data Preparation
# Inspect the dataset
print(titanic.head())
print(titanic.info())
print(titanic.describe())

# Handling missing values
titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)
titanic.drop(columns=['deck'], inplace=True)

# Step 3: Exploratory Data Analysis (EDA)
# Descriptive statistics
print(titanic.describe())

# Analyzing survival rate by different features
print(titanic.groupby('sex')['survived'].mean())
print(titanic.groupby('class')['survived'].mean())
print(titanic.groupby('embarked')['survived'].mean())

# Step 4: Data Visualization
# Visualization 1: Survival rate by sex
plt.figure(figsize=(10, 6))
sns.barplot(x='sex', y='survived', data=titanic)
plt.title('Survival Rate by Sex')
plt.ylabel('Survival Rate')
plt.xlabel('Sex')
plt.show()

# Visualization 2: Survival rate by passenger class
plt.figure(figsize=(10, 6))
sns.barplot(x='class', y='survived', data=titanic)
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.xlabel('Class')
plt.show()

# Visualization 3: Age distribution by survival
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic, x='age', hue='survived', multiple='stack')
plt.title('Age Distribution by Survival')
plt.ylabel('Count')
plt.xlabel('Age')
plt.show()

# Visualization 4: Survival rate by embarkation point
plt.figure(figsize=(10, 6))
sns.barplot(x='embarked', y='survived', data=titanic)
plt.title('Survival Rate by Embarkation Point')
plt.ylabel('Survival Rate')
plt.xlabel('Embarked')
plt.show()

# Visualization 5: Pairplot of numerical features
plt.figure(figsize=(12, 8))
sns.pairplot(titanic[['age', 'fare', 'survived']], hue='survived', palette='coolwarm')
plt.suptitle('Pairplot of Age, Fare and Survival', y=1.02)
plt.show()
