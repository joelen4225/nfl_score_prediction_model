# CatBoost
# Average baseline error: 7.35
# Mean Absolute Error: 8.17 degrees

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from data_prep import *

# Defining Testing and Training Data
train_df = model_final_df[model_final_df['season'] < 2023]
test_df = model_final_df[model_final_df['season'] >= 2023]

# Defining X & Y Variables
X_train = train_df.drop(columns=['season', 'week', 'team_abbr', 'opponent', 'team_score'])
y_train = train_df['team_score']
X_test = test_df.drop(columns=['season', 'week', 'team_abbr', 'opponent', 'team_score'])
y_test = test_df['team_score']

# Using a Pipeline to Account for Missing Values
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', CatBoostRegressor(iterations=100, random_seed=42, verbose=0))
])

# Finding Out My Baseline Using Implied Score
baseline_preds = test_df['implied_score']
baseline_errors = abs(baseline_preds - y_test)
print('Average baseline error:', round(np.mean(baseline_errors), 2))

# Creating and Fitting the Model
pipeline.fit(X_train, y_train)

# Predictions
y_pred_test = pipeline.predict(X_test)

# Evaluating the Model
errors = abs(y_pred_test - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')