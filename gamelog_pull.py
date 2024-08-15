# Importing Packages
import pandas as pd
import nflscraPy as nfl

# Naming Year Variable for Data Import
years = list(range(2000, 2024))

# Initializing an Empty Data Frame List
dfs = []

# Looping Through Each Year and Fetching Games
for year in years:
    df = nfl._gamelogs(year)
    df['season'] = year 
    print(df.head())
    if year < 2021:
        df = df[df['week'].between(1, 16)]
    else:
        df = df[df['week'].between(1, 17)]
    df = df[['season', 'week', 'tm_alias', 'opp_alias', 'tm_score', 'opp_score']]
    dfs.append(df) 

# Concatenate All Data Frames into 1
game_logs = pd.concat(dfs, ignore_index=True)
game_logs.to_csv('data/game_logs_2000_2023.csv')