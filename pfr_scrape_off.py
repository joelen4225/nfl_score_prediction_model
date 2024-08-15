# This is Based Off of a YouTube Video by Kerry Sports Analyst
# https://www.youtube.com/watch?v=2JDR6jv0fGA

# Import Libraries
import pandas as pd
import numpy as np
import random
import time
import urllib

# Create a List of Seasons and Verify with #
seasons = [str(season) for season in range(2016, 2024)]
print(f'Number of Seasons = {len(seasons)}')

# Create a List of Team Abbreviations and Verify with #
team_abbr = ['crd', 'atl', 'rav', 'buf', 'car', 'chi', 'cin', 'cle', 'dal', 'den', 'det', 'gnb', 'htx', 'clt', 'jax', 'kan',
             'sdg', 'ram', 'rai', 'mia', 'min', 'nwe', 'nor', 'nyg', 'nyj', 'phi', 'pit', 'sea', 'sfo', 'tam', 'oti', 'was']
print(f'Number of Teams = {len(team_abbr)}')

nfl_df = pd.DataFrame()

## Iterate Through Seasons
for season in seasons:
    ## Iterate Through Teams
    for team in team_abbr:
        url = 'https://www.pro-football-reference.com/teams/' + team + '/' + season +'/gamelog/'
        print(url)

        off_df = pd.read_html(url, header = 1, attrs = {'id':'gamelog' + season})[0]
        def_df = pd.read_html(url, header = 1, attrs = {'id':'gamelog_opp' + season})[0]

        team_df = pd.concat([off_df, def_df], axis = 1)
        team_df.insert(loc = 0, column = "season", value = season)
        team_df.insert(loc = 2, column = "Team", value = team.upper())

        nfl_df = pd.concat([nfl_df, team_df], ignore_index = True)

        time.sleep(random.randint(4, 5))

# Creating a CSV File
nfl_df.to_csv('nfl_gamelogs_2016_2023.csv', index = False)
