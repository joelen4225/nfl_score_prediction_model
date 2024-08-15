# Stacking with Linear Regression & Random Forest
# Average baseline error: 7.35
# Mean Absolute Error: 7.43 degrees

import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data_prep import *

# Defining Testing and Training Data
train_df = model_final_df[model_final_df['season'] < 2023]
test_df = model_final_df[model_final_df['season'] >= 2023]

# Defining X & Y Variables
X_train = train_df.drop(columns=['season', 'week', 'team_abbr', 'opponent', 'team_score'])
y_train = train_df['team_score']
X_test = test_df.drop(columns=['season', 'week', 'team_abbr', 'opponent', 'team_score'])
y_test = test_df['team_score']

# Normalizing the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Defining Base Models
base_models = [
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('linear_regression', LinearRegression())
]

# Defining Meta-Models
meta_model = LinearRegression()

# Creating the Stack Regressor
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model
)

# Finding Out My Baseline Using Implied Score
baseline_preds = test_df['implied_score']
baseline_errors = abs(baseline_preds - y_test)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Using a Pipeline to Account for Missing Values
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('stacking', stacking_model)
])

# Creating and Fitting the Model
pipeline.fit(X_train, y_train)

# Predictions
y_pred_test = pipeline.predict(X_test)

# Evaluating the Model
errors = abs(y_pred_test - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')