# Voting Regression (Random Forest & LightGBM) & RandomizedSearchCV
# Average baseline error: 7.35
# Mean Absolute Error:  7.47 degrees

import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
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

# Ensure No Missing Values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Defining the Models
lgbm = LGBMRegressor(verbose=-1)  # Suppress LightGBM warnings
rf = RandomForestRegressor()

# Creating the Voting Regressor
voting_regressor = VotingRegressor([('lgbm', lgbm), ('rf', rf)])

# Using a Pipeline to Account for Missing Values
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('voting_regressor', voting_regressor)
])

# Defining the Random Forest model and Parameter Grid
param_grid = {
    'voting_regressor__lgbm__n_estimators': randint(50, 150),  # Reduced range
    'voting_regressor__lgbm__learning_rate': uniform(0.01, 0.1),  # Reduced range
    'voting_regressor__lgbm__num_leaves': randint(20, 40),  # Reduced range
    'voting_regressor__lgbm__max_depth': randint(5, 10),
    'voting_regressor__lgbm__min_child_samples': randint(5, 20),  # Reduced range
    'voting_regressor__lgbm__min_split_gain': uniform(0.0, 0.05),  # Reduced range
    'voting_regressor__rf__n_estimators': randint(50, 150),  # Reduced range
    'voting_regressor__rf__max_features': ['sqrt', 'log2', None],
    'voting_regressor__rf__min_samples_split': randint(2, 10),
    'voting_regressor__rf__min_samples_leaf': randint(1, 5),
    'voting_regressor__rf__max_depth': randint(5, 10)  # Adjusted range
}

# Use RandomizedSearchCV with the Voting Regressor
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=30, cv=3, n_jobs=-1, verbose=1, random_state=42)  # Reduced n_iter and verbose

# Fitting the RandomizedSearchCV
random_search.fit(X_train, y_train)

# Predictions
y_pred_test = random_search.best_estimator_.predict(X_test)

# Evaluating the Model
errors = abs(y_pred_test - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print('Best Parameters:', random_search.best_params_)
