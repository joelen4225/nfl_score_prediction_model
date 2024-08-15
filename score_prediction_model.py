# For my final model, I am using a Voting Regression that combines Random Forest & LightGBM
# The original model can be found at 'modeling_attempts/model_v14.py'
# The starting MAE was 7.47
# This code looks to optimize the MAE and ultimately perform better than implied score from sportsbooks (7.35)
# Current Best MAE: 7.37

from datetime import datetime
start_time = datetime.now()
print("Start Time:", start_time)

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import random
import concurrent.futures
from data_prep import *

train_df = model_final_df[model_final_df['season'] < 2023]
test_df = model_final_df[model_final_df['season'] >= 2023]

# Defining X & Y Variables
X_train = train_df.drop(columns=['season', 'week', 'team_abbr', 'opponent', 'team_score'])
y_train = train_df['team_score']
X_test = test_df.drop(columns=['season', 'week', 'team_abbr', 'opponent', 'team_score'])
y_test = test_df['team_score']

# Ensuring No Missing Values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Feature Selection using Feature Importance from RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train_scaled, y_train)
importance = rf.feature_importances_

# Selecting the Top Features Based on Importance
num_top_features = 205
indices = np.argsort(importance)[-num_top_features:]
X_train_selected = X_train_scaled[:, indices]
X_test_selected = X_test_scaled[:, indices]

# Defining Models
lgbm = LGBMRegressor(verbose=-1)

# Creating the Voting Regressor
voting_regressor = VotingRegressor([('lgbm', lgbm), ('rf', rf)])

# Using a Pipeline to Account for Missing Values and Scaling
pipeline = Pipeline([
    ('voting_regressor', voting_regressor)
])

# Defining Parameter Grid
param_grid = {
    'voting_regressor__lgbm__n_estimators': randint(50, 150),
    'voting_regressor__lgbm__learning_rate': uniform(0.01, 0.1),
    'voting_regressor__lgbm__num_leaves': randint(20, 40),
    'voting_regressor__lgbm__max_depth': randint(5, 10),
    'voting_regressor__lgbm__min_child_samples': randint(5, 20),
    'voting_regressor__lgbm__min_split_gain': uniform(0.0, 0.05),
    'voting_regressor__rf__n_estimators': randint(50, 150),
    'voting_regressor__rf__max_features': ['sqrt', 'log2', None],
    'voting_regressor__rf__min_samples_split': randint(2, 10),
    'voting_regressor__rf__min_samples_leaf': randint(1, 5),
    'voting_regressor__rf__max_depth': randint(5, 10)
}

# Using RandomizedSearchCV with the Voting Regressor
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=30, cv=3, n_jobs=-1, verbose=1, random_state=42)

# Fitting the RandomizedSearchCV
random_search.fit(X_train_selected, y_train)

# Predictions
y_pred_test = random_search.best_estimator_.predict(X_test_selected)

test_df = test_df.copy()
test_df.loc[:, 'predicted_score'] = y_pred_test



# Creating a Game ID to Match Games
def generate_game_id(row):
    team1, team2 = sorted([row['team_abbr'], row['opponent']])
    return f"{row['season']}_{row['week']}_{team1}_{team2}"

test_df.loc[:, 'game_id'] = test_df.apply(generate_game_id, axis=1)
test_df.loc[:, 'ml'] = np.where(test_df['spread'] < 1, test_df['team_abbr'], test_df['opponent'])

# Creating Consolidated Game Data to Evaluate Outcomes
game_data = []
for game_id, group in test_df.groupby('game_id'):
    if len(group) != 2:
        continue

    # Extracting Home and Away Rows
    home_row = group.iloc[0]
    away_row = group.iloc[1]
    if home_row['team_abbr'] == away_row['opponent']:
        home_row, away_row = home_row, away_row
    elif away_row['team_abbr'] == home_row['opponent']:
        pass
    else:
        continue

    # Calculating New Values
    home_team = home_row['team_abbr']
    away_team = home_row['opponent']
    total = home_row['total']
    spread = home_row['spread']
    ml = home_row['ml']
    home_implied_score = home_row['implied_score']
    away_implied_score = away_row['implied_score']
    home_score = home_row['team_score']
    away_score = away_row['team_score']
    home_predicted_score = home_row['predicted_score']
    away_predicted_score = away_row['predicted_score']
    predicted_spread = away_predicted_score - home_predicted_score
    actual_spread = away_score - home_score
    predicted_total = away_predicted_score + home_predicted_score
    actual_total = away_score + home_score
    if home_predicted_score > away_predicted_score:
        predicted_winner = home_team
    elif away_predicted_score > home_predicted_score:
        predicted_winner = away_team
    else:
        predicted_winner = ''
    if home_score > away_score:
        actual_winner = home_team
    elif away_score > home_score:
        actual_winner = away_team
    else:
        actual_winner = ''
    spread_won = ((predicted_spread < spread and actual_spread < spread) or 
               (predicted_spread > spread and actual_spread > spread))
    total_won = ((predicted_total < total and actual_total < total) or 
               (predicted_spread > total and actual_spread > total))
    ml_won = (predicted_winner == ml) and (actual_winner == ml)
    
    game_data.append({
        'season': home_row['season'],
        'week': home_row['week'],
        'home_team': home_team,
        'away_team': away_team,
        'total': total,
        'spread': spread,
        'ml': ml,
        'home_implied_score': home_implied_score,
        'away_implied_score': away_implied_score,
        'home_score': home_score,
        'away_score': away_score,
        'home_predicted_score': home_predicted_score,
        'away_predicted_score': away_predicted_score,
        'predicted_spread': predicted_spread,
        'actual_spread': actual_spread,
        'predicted_total': predicted_total,
        'actual_total': actual_total,
        'predicted_winner': predicted_winner,
        'actual_winner': actual_winner,
        'spread_won': spread_won,
        'total_won': total_won,
        'ml_won': ml_won
    })

# Creating the Consolidated Game Data DF
consolidated_df = pd.DataFrame(game_data)

spread_won_pct = consolidated_df['spread_won'].sum() / len(consolidated_df)
total_won_pct = consolidated_df['total_won'].sum() / len(consolidated_df)
ml_won_pct = consolidated_df['ml_won'].sum() / len(consolidated_df)

print(f"% spread won: {spread_won_pct:.2%}")
print(f"% total won: {total_won_pct:.2%}")
print(f"% ml won: {ml_won_pct:.2%}")

end_time = datetime.now()
print("End Time:", end_time)
print("Time Elapsed:", end_time - start_time)

# Saving to CSV
consolidated_df.to_csv('outcomes/consolidated_game_data.csv', index=False)