import pandas as pd
from gamelog_pull import *

# Create Empty Dictionaries to Score Game Frequencies for Each Team
score_frequencies = {}

# Iterate through each row of the DataFrame
for _, row in game_logs.iterrows():
    for team, score in [(row['tm_alias'], row['tm_score']), (row['opp_alias'], row['opp_score'])]:
        if team not in score_frequencies:
            score_frequencies[team] = {}
        if score not in score_frequencies[team]:
            score_frequencies[team][score] = 0
        score_frequencies[team][score] += 1

# Converting the Dictionary to a Data Frame with Easier Readability
data = []
for team, scores in score_frequencies.items():
    total_games = sum(scores.values())
    for score, count in scores.items():
        frequency = count / total_games
        data.append({'team': team, 'score': score, 'frequency': frequency})

frequency_df = pd.DataFrame(data)

frequency_df.to_csv('data/team_score_frequencies.csv', index=False)
