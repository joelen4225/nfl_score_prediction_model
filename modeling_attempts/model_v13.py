# Random Forest & RandomizedSearchCV
# Average baseline error: 7.35
# Mean Absolute Error:  7.47 degrees

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from data_prep import *

# Defining Testing and Training Data
train_df = model_final_df[model_final_df['season'] < 2023]
test_df = model_final_df[model_final_df['season'] >= 2023]

# Defining X & Y Variables
X_train = train_df.drop(columns=['season', 'week', 'team_abbr', 'opponent', 'team_score'])
y_train = train_df['team_score']
X_test = test_df.drop(columns=['season', 'week', 'team_abbr', 'opponent', 'team_score'])
y_test = test_df['team_score']

# Defining the Random Forest model and Parameter Grid
model = RandomForestRegressor(random_state=42)
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_features': ['sqrt', 'log2', None],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Creating a Pipeline with Imputer and StandardScaler
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
    ('scaler', StandardScaler()),  # Normalize the data
    ('model', model)
])

# Setting up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1, error_score='raise')

# Fitting the RandomizedSearchCV
random_search.fit(X_train, y_train)

# Predictions
y_pred_test = random_search.best_estimator_.predict(X_test)

# Evaluating the Model
errors = abs(y_pred_test - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print('Best Parameters:', random_search.best_params_)
