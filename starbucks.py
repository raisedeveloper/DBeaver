import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv(',/dataset/StarbucksSurvey_csv')

x = dataset[['Gender', 'Age', 'purchase']]
y = dataset['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, X_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.pre
