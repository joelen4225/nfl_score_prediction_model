# Importing Packages
import pandas as pd
import numpy as np
from next_gen_pull import *
from injury_pull import *

# A Function that Fills Missing Values with Current Season or Most Recent Season Averages
def fill_missing_values(row, column, season_median, most_recent_season_median):
    if pd.isna(row[column]):
        if row['season'] in season_median.index and not pd.isna(season_median.loc[row['season'], column]):
            return season_median.loc[row['season'], column]
        else:
            return most_recent_season_median[column]
    else:
        return row[column]

# Cleaning Passing Data
## Clearing Out Non-Regular Season Games
df_passing_reg = df_passing[(df_passing['week'] != 0) & (df_passing['season_type'] == 'REG')]
## Singling Out the 'Main QB' Using # of Pass Attempts
idx = df_passing_reg.groupby(['season', 'week', 'team_abbr'])['attempts'].idxmax()
df_passing_reg_main = df_passing_reg.loc[idx]
df_passing_reg_main = df_passing_reg_main.reset_index(drop=True)
## Dropping Unneeded Columns
df_passing_filtered = df_passing_reg_main.drop(columns = ["season_type", "player_display_name", "player_position",
                                                          "player_gsis_id", "player_first_name", "player_last_name",
                                                          "player_jersey_number", "player_short_name"])
## Filling Missing Values with League Season Averages
passing_columns = ["avg_time_to_throw", "avg_completed_air_yards", "avg_intended_air_yards", "avg_air_yards_differential",
                   "aggressiveness", "max_completed_air_distance", "avg_air_yards_to_sticks", "attempts", "pass_yards",
                   "pass_touchdowns", "interceptions", "passer_rating", "completions", "completion_percentage", 
                   "expected_completion_percentage", "completion_percentage_above_expectation", "avg_air_distance", "max_air_distance"]
passing_season_median = df_passing_filtered.groupby('season')[passing_columns].median()
pass_most_recent_season_median = passing_season_median.loc[passing_season_median.index != df_passing_filtered['season'].max()].iloc[-1]
for column in passing_columns:
    df_passing_filtered[column] = df_passing_filtered.apply(fill_missing_values, axis=1, args=(column, 
                                                                                               passing_season_median, 
                                                                                               pass_most_recent_season_median))



# Cleaning Receiving Data
## Clearing Out Non-Regular Season Games
df_receiving_reg = df_receiving[(df_receiving['week'] != 0) & (df_receiving['season_type'] == 'REG')]
## Singling Out the 'Main WR' Using # of Receptions
idx = df_receiving_reg.groupby(['season', 'week', 'team_abbr'])['receptions'].idxmax()
df_receiving_reg_main = df_receiving_reg.loc[idx]
df_receiving_reg_main = df_receiving_reg_main.reset_index(drop=True)
## Dropping Unneeded Columns
df_receiving_filtered = df_receiving_reg_main.drop(columns = ["season_type", "player_display_name", "player_position",
                                                          "player_gsis_id", "player_first_name", "player_last_name",
                                                          "player_jersey_number", "player_short_name"])
## Filling Missing Values with League Season Averages
receiving_columns = ["avg_cushion", "avg_separation", "avg_intended_air_yards", "percent_share_of_intended_air_yards", "receptions", 
                     "targets", "catch_percentage", "yards", "rec_touchdowns", "avg_yac", "avg_expected_yac", "avg_yac_above_expectation"]
receiving_season_median = df_receiving_filtered.groupby('season')[receiving_columns].median()
rec_most_recent_season_median = receiving_season_median.loc[receiving_season_median.index != df_receiving_filtered['season'].max()].iloc[-1]
for column in receiving_columns:
    df_receiving_filtered[column] = df_receiving_filtered.apply(fill_missing_values, axis=1, args=(column, 
                                                                                               receiving_season_median, 
                                                                                               rec_most_recent_season_median))

# Cleaning Rushing Data
## Clearing Out Non-Regular Season Games
df_rushing_reg = df_rushing[(df_rushing['week'] != 0) & (df_rushing['season_type'] == 'REG')]
## Singling Out the 'Main RB' Using # of Rush Attempts
idx = df_rushing_reg.groupby(['season', 'week', 'team_abbr'])['rush_attempts'].idxmax()
df_rushing_reg_main = df_rushing_reg.loc[idx]
df_rushing_reg_main = df_rushing_reg_main.reset_index(drop=True)
## Dropping Unneeded Columns
df_rushing_filtered = df_rushing_reg_main.drop(columns = ["season_type", "player_display_name", "player_position",
                                                          "player_gsis_id", "player_first_name", "player_last_name",
                                                          "player_jersey_number", "player_short_name"])
## Filling Missing Values with League Season Averages
rushing_columns = ["efficiency", "percent_attempts_gte_eight_defenders", "avg_time_to_los", "rush_attempts", "rush_yards",
                   "expected_rush_yards", "rush_yards_over_expected", "avg_rush_yards", "rush_yards_over_expected_per_att",
                   "rush_pct_over_expected", "rush_touchdowns"]
rushing_season_median = df_rushing_filtered.groupby('season')[rushing_columns].median()
rush_most_recent_season_median = rushing_season_median.loc[rushing_season_median.index != df_rushing_filtered['season'].max()].iloc[-1]
for column in rushing_columns:
    df_rushing_filtered[column] = df_rushing_filtered.apply(fill_missing_values, axis=1, args=(column, 
                                                                                               rushing_season_median, 
                                                                                               rush_most_recent_season_median))
    
# Merging Passing, Receiving, and Rushing Data for Weekly Offensive Stats
merged_off_df = df_passing_filtered.merge(df_receiving_filtered, on=['season', 'week', 'team_abbr'], how='left')
merged_off_df = merged_off_df.merge(df_rushing_filtered, on=['season', 'week', 'team_abbr'], how='left')
## Filling Missing Values with League Season Averages
merged_columns = ["avg_time_to_throw", "avg_completed_air_yards", "avg_intended_air_yards_x", "avg_air_yards_differential",
                   "aggressiveness", "max_completed_air_distance", "avg_air_yards_to_sticks", "attempts", "pass_yards",
                   "pass_touchdowns", "interceptions", "passer_rating", "completions", "completion_percentage", 
                   "expected_completion_percentage", "completion_percentage_above_expectation", "avg_air_distance", "max_air_distance",
                   "avg_cushion", "avg_separation", "avg_intended_air_yards_y", "percent_share_of_intended_air_yards", "receptions", 
                   "targets", "catch_percentage", "yards", "rec_touchdowns", "avg_yac", "avg_expected_yac", "avg_yac_above_expectation",
                   "efficiency", "percent_attempts_gte_eight_defenders", "avg_time_to_los", "rush_attempts", "rush_yards",
                   "expected_rush_yards", "rush_yards_over_expected", "avg_rush_yards", "rush_yards_over_expected_per_att",
                   "rush_pct_over_expected", "rush_touchdowns"]
season_median = merged_off_df.groupby('season')[merged_columns].median()
most_recent_season_median = season_median.loc[season_median.index != merged_off_df['season'].max()].iloc[-1]
for column in merged_columns:
    merged_off_df[column] = merged_off_df.apply(fill_missing_values, axis=1, args=(column, 
                                                                                    season_median, 
                                                                                    most_recent_season_median))
    
## Model DF Version 1 Was Created Here
# Creating Score Predictions Using Total & Spread
## Uploaded the Downloaded CSV File
weekly_lines_df = pd.read_csv('data/spreadspoke_scores.csv')
## Filtering Out for Games Between 2016 and 2023 & Regular Season
weekly_lines_df = weekly_lines_df[(weekly_lines_df['schedule_season'] >= 2016) & (weekly_lines_df['schedule_season'] <= 2023)]
weekly_lines_df = weekly_lines_df[(weekly_lines_df['schedule_week'] != 0) & (weekly_lines_df['schedule_playoff'] == False)]
## Renaming Columns for Merge
weekly_lines_df = weekly_lines_df.rename(columns = {'schedule_season': 'season', 'schedule_week': 'week',
                                                    'spread_favorite': 'spread', 'over_under_line': 'total'})
## Renaming Certain Teams for Value
team_rename_dict = {
    "Kansas City Chiefs": "KC",
    "Philadelphia Eagles": "PHI",
    "Atlanta Falcons": "ATL",
    "Buffalo Bills": "BUF",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Detroit Lions": "DET",
    "Indianapolis Colts": "IND",
    "Los Angeles Chargers": "LAC",
    "Miami Dolphins": "MIA",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "San Francisco 49ers": "SF",
    "Arizona Cardinals": "ARI",
    "Baltimore Ravens": "BAL",
    "Carolina Panthers": "CAR",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Jacksonville Jaguars": "JAX",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "New York Jets": "NYJ",
    "Las Vegas Raiders": "LV",
    "Los Angeles Rams": "LAR",
    "Pittsburgh Steelers": "PIT",
    "Washington Football Team": "WAS",
    "Oakland Raiders": "LV",
    "Washington Redskins": "WAS",
    "San Diego Chargers": "LAC",
    "St. Louis Rams": "LAR",
    "Tennessee Oilers": "TEN",
    "Houston Oilers": "TEN",
    "Los Angeles Raiders": "LV",
    "Phoenix Cardinals": "ARI",
    "St. Louis Cardinals": "ARI",
    "Baltimore Colts": "IND",
    "Boston Patriots": "NE"
}
weekly_lines_df['team_away'] = weekly_lines_df['team_away'].replace(team_rename_dict)
weekly_lines_df['team_home'] = weekly_lines_df['team_home'].replace(team_rename_dict)
## Altering the Data to Have Two Rows for Each Game
transformed_rows = []
for index, row in weekly_lines_df.iterrows():
    home_row = {
        'season': row['season'],
        'week': row['week'],
        'team_abbr': row['team_home'],
        'spread': row['spread'] if row['team_favorite_id'] == row['team_home'] else -row['spread'],
        'total': row['total']
    }
    away_row = {
        'season': row['season'],
        'week': row['week'],
        'team_abbr': row['team_away'],
        'spread': row['spread'] if row['team_favorite_id'] == row['team_away'] else -row['spread'],
        'total': row['total']
    }
    transformed_rows.append(home_row)
    transformed_rows.append(away_row)
sportsbook_df = pd.DataFrame(transformed_rows)
sportsbook_df['total'] = sportsbook_df['total'].astype(float)
## Performing Calculations
sportsbook_df['implied_score'] = (sportsbook_df['total'] / 2 - sportsbook_df['spread'] / 2).round()
sportsbook_df.to_csv("data/sportsbook.csv")
## Merging with Offensive Statistics
sportsbook_df['week'] = sportsbook_df['week'].astype(int)
model_df = pd.merge(merged_off_df, sportsbook_df, on=['season', 'week', 'team_abbr'], how='left')

## Model DF Version 2 Was Created Here

# Cleaning the Gamelogs Data
## Uploading the Created CSV File
nfl_df = pd.read_csv('data/nfl_gamelogs_2016_2023.csv')
## Dropping Unneeded Columns
duplicate_columns = nfl_df.columns[38:74]

nfl_df = nfl_df.drop(columns= duplicate_columns)
nfl_df_updated = nfl_df.drop(columns = ["Day", "Date", "Unnamed: 3", "Unnamed: 4", "OT", "Unnamed: 6", "ToP"])
## Updating the 'Team' Column
nfl_df_updated = nfl_df_updated.rename(columns={'Team': 'team_abbr'})
replacement_dict = {
    'CRD': 'ARI',
    'RAV': 'BAL',
    'GNB': 'GB',
    'HTX': 'HOU',
    'CLT': 'IND',
    'KAN': 'KC',
    'SDG': 'LAC',
    'RAM': 'LAR',
    'RAI': 'LV',
    'NWE': 'NE',
    'NOR': 'NO',
    'SFO': 'SF',
    'TAM': 'TB',
    'OTI': 'TEN'
}
nfl_df_updated['team_abbr'] = nfl_df_updated['team_abbr'].replace(replacement_dict)
## Renaming Columns for Easier Detection
rename_dict = {
    'Week': 'week',
    'Opp': 'opponent',
    'Tm': 'team_score',
    'Opp.1': 'opp_score',
    'Cmp': 'team_completions',
    'Att': 'team_attempts',
    'Yds': 'team_passing_yards',
    'TD': 'team_passing_touchdowns',
    'Int': 'team_interceptions',
    'Sk': 'team_sacks',
    'Yds.1': 'team_sack_yards_lost',
    'Y/A': 'team_yards_per_attempt',
    'NY/A': 'team_net_yards_per_attempt',
    'Cmp%': 'team_completion_percentage',
    'Rate': 'team_passer_rating',
    'Att.1': 'team_rushing_attempts',
    'Yds.2': 'team_rushing_yards',
    'Y/A.1': 'team_rushing_yards_per_attempt',
    'TD.1': 'team_rushing_touchdowns',
    'FGM': 'team_field_goals_made',
    'FGA': 'team_field_goals_attempted',
    'XPM': 'team_extra_points_made',
    'XPA': 'team_extra_points_attempted',
    'Pnt': 'team_points',
    'Yds.3': 'team_total_yards',
    '3DConv': 'team_3rd_down_conversions',
    '3DAtt': 'team_3rd_down_attempts',
    '4DConv': 'team_4th_down_conversions',
    '4DAtt': 'team_4th_down_attempts'
}
nfl_df_updated = nfl_df_updated.rename(columns=rename_dict)
## Revaluing Opponent Teams
opponent_rename_dict = {
    'New England Patriots': 'NE',
    'Tampa Bay Buccaneers': 'TB',
    'Buffalo Bills': 'BUF',
    'Los Angeles Rams': 'LAR',
    'San Francisco 49ers': 'SF',
    'New York Jets': 'NYJ',
    'Seattle Seahawks': 'SEA',
    'Carolina Panthers': 'CAR',
    'Minnesota Vikings': 'MIN',
    'Atlanta Falcons': 'ATL',
    'Washington Redskins': 'WAS',
    'Miami Dolphins': 'MIA',
    'New Orleans Saints': 'NO',
    'Oakland Raiders': 'LV',
    'Denver Broncos': 'DEN',
    'San Diego Chargers': 'LAC',
    'Green Bay Packers': 'GB',
    'Philadelphia Eagles': 'PHI',
    'Arizona Cardinals': 'ARI',
    'Kansas City Chiefs': 'KC',
    'Cleveland Browns': 'CLE',
    'Jacksonville Jaguars': 'JAX',
    'New York Giants': 'NYG',
    'Pittsburgh Steelers': 'PIT',
    'Dallas Cowboys': 'DAL',
    'Cincinnati Bengals': 'CIN',
    'Baltimore Ravens': 'BAL',
    'Houston Texans': 'HOU',
    'Detroit Lions': 'DET',
    'Indianapolis Colts': 'IND',
    'Tennessee Titans': 'TEN',
    'Chicago Bears': 'CHI',
    'Los Angeles Chargers': 'LAC',
    'Washington Football Team': 'WAS',
    'Las Vegas Raiders': 'LV',
    'Washington Commanders': 'WAS'
}
nfl_df_updated['opponent'] = nfl_df_updated['opponent'].replace(opponent_rename_dict)
## Merging with Offensive Statistics and Sportsbook Lines
model_df = pd.merge(nfl_df_updated, model_df, on=['season', 'week', 'team_abbr'], how='left')

## Model DF Version 3 Was Created Here

# Cleaning Up the Defensive Data (2018 - 2023)
## Importing Data
defense_df = pd.read_csv('data/defensive_data.csv')
## Clearing Out Non-Regular Season Games
defense_reg_df = defense_df[(defense_df['week'] != 0) & (defense_df['game_type'] == 'REG')]
## Assigning Which Column Names to Sum
sum_columns = [
    'def_ints', 'def_targets', 'def_completions_allowed', 'def_yards_allowed',
    'def_receiving_td_allowed', 'def_times_blitzed', 'def_times_hurried',
    'def_times_hitqb', 'def_sacks', 'def_pressures', 'def_tackles_combined',
    'def_missed_tackles', 'def_air_yards_completed', 'def_yards_after_catch'
]
## Assigning Which Column Names to Average
avg_columns = [
    'def_completion_pct', 'def_yards_allowed_per_cmp', 'def_yards_allowed_per_tgt',
    'def_passer_rating_allowed', 'def_adot', 'def_missed_tackle_pct'
]
## Sum & Average the Columns
defense_reg_df = defense_reg_df.rename(columns={'team': 'team_abbr'})
team_sum_stats = defense_reg_df.groupby(['game_id', 'team_abbr'])[sum_columns].sum()
team_avg_stats = defense_reg_df.groupby(['game_id', 'team_abbr'])[avg_columns].mean()
## Combine into One Data Frame
team_stats = pd.concat([team_sum_stats, team_avg_stats], axis=1).reset_index()
## Filtering Out Original Columns and Extracting Duplicates
columns_to_keep = ['game_id', 'season', 'week', 'team_abbr', 'opponent']
defense_final_df = defense_reg_df[columns_to_keep].drop_duplicates(subset=['game_id', 'team_abbr']).set_index(['game_id', 'team_abbr'])
## Merging Calculated Stats with Identifying Columns
defense_final_df = defense_final_df.merge(team_stats, left_index=True, right_on=['game_id', 'team_abbr']).reset_index(drop=True)
## Revaluing Certain team_abbr
rename_dict = {
    'OAK': "LV",
    'LA': 'LAR'
}
defense_final_df['team_abbr'] = defense_final_df['team_abbr'].replace(rename_dict)
## Merging with Offensive Statistics, Sportsbook Lines, and Gamelog Data
defense_final_df = defense_final_df.drop(columns = ["game_id"])

# Cleaning Up the Injury Data
## Updating the 'team' Column
injury_df['team'] = injury_df['team'].replace({'LA': 'LAR', 'OAK': 'LV', 'SD': 'LAC'})
## Using Only Regular Season Data
condition_2016_2020 = (injury_df['season'].between(2016, 2020)) & (injury_df['week'].between(1, 17))
condition_2021_2023 = (injury_df['season'].between(2021, 2023)) & (injury_df['week'].between(1, 17))
combined_condition = condition_2016_2020 | condition_2021_2023
injury_df = injury_df[combined_condition]
## Getting Injury Totals
def aggregate_status_counts(group):
    group = group[['report_status', 'practice_status']]
    return pd.Series({
        'players_out_total': (group['report_status'] == 'Out').sum(),
        'players_questionable_total': (group['report_status'] == 'Questionable').sum(),
        'players_doubtful_total': (group['report_status'] == 'Doubtful').sum(),
        'players_full_total': (group['practice_status'] == 'Full Participation in Practice').sum(),
        'players_none_total': (group['practice_status'] == 'Did Not Participate In Practice').sum(),
        'players_limit_total': (group['practice_status'] == 'Limited Participation in Practice').sum()
    })
team_injury_df = injury_df.groupby(['season', 'week', 'team'])
team_injury_df = team_injury_df.apply(lambda g: aggregate_status_counts(g)).reset_index()
## Merging with the Rest of the Data
team_injury_df = team_injury_df.rename(columns={'team': 'team_abbr'})
model_df = pd.merge(model_df, team_injury_df, on=['season', 'week', 'team_abbr'], how='left')

# Creating Average Batches for Model
## Creating a Function to Calculate Rolling Averages for Each Team
def calculate_rolling_averages(df, team, season, week, columns, windows):
    ## Filter Data for the Team Before the Given Game
    team_data = df[(df['team_abbr'] == team) & 
                   ((df['season'] < season) | ((df['season'] == season) & (df['week'] < week)))]
    
    ## Sort By Season and Week
    team_data = team_data.sort_values(by=['season', 'week'])
    
    averages = {}
    for window in windows:
        ## Calculate Rolling Averages
        if len(team_data) < window:
            ## Use Available Games if Less Than Window Size
            rolling_avg = team_data[columns].rolling(window=len(team_data)).mean().iloc[-1] if not team_data.empty else pd.Series([None]*len(columns), index=columns)
        else:
            rolling_avg = team_data[columns].rolling(window=window).mean().iloc[-1]
        
        for column in columns:
            averages[f"{column}_L{window}"] = rolling_avg[column]
    
    ## Entire Current Season
    season_data = team_data[team_data['season'] == season]
    season_avg = season_data[columns].mean()
    for column in columns:
        averages[f"{column}_Szn"] = season_avg[column]
    
    ## Entire Previous Season
    previous_season = season - 1
    previous_season_data = df[(df['season'] == previous_season) & (df['team_abbr'] == team)]
    previous_season_avg = previous_season_data[columns].mean()
    for column in columns:
        averages[f"{column}_LSzn"] = previous_season_avg[column]

    return averages
## Defining Columns and Windows to Calculate On
columns_to_exclude = ['total', 'spread', 'implied_score', 'players_out_total', 'players_questionable_total', 
                      'players_doubtful_total', 'players_full_total','players_none_total', 'players_limit_total']
columns = [col for col in model_df.columns[4:] if col not in columns_to_exclude]
def_columns = defense_final_df.columns[4:]
columns_delete = [col for col in model_df.columns[5:] if col not in columns_to_exclude]
def_columns_del = defense_final_df.columns[4:]
windows = [1, 5, 10]
## Iterating Over Each Row and Creating Average Batches
average_batches = []
for index, row in model_df.iterrows():
    team = row['team_abbr']
    season = row['season']
    week = row['week']
    averages = calculate_rolling_averages(model_df, team, season, week, columns, windows)
    average_batches.append(averages)
def_average_batches = []
for index, row in defense_final_df.iterrows():
    team = row['team_abbr']
    season = row['season']
    week = row['week']
    averages = calculate_rolling_averages(defense_final_df, team, season, week, def_columns, windows)
    def_average_batches.append(averages)
## Creating a Data Frame and Merging with Model
average_batches_df = pd.DataFrame(average_batches)
def_average_batches_df = pd.DataFrame(def_average_batches)
model_final_df = pd.concat([model_df, average_batches_df], axis = 1)
def_final_df = pd.concat([defense_final_df, def_average_batches_df], axis = 1)
## Dropping In-Game Statistics
model_final_df = model_final_df.drop(columns = columns_delete)
def_final_df = def_final_df.drop(columns = def_columns_del)
## Filling Missing Values with League Season Averages
season_median = model_final_df.groupby('season')[model_final_df.columns[4:]].median()
def_season_median = def_final_df.groupby('season')[def_final_df.columns[4:]].median()
most_recent_season_median = season_median.loc[season_median.index != model_final_df['season'].max()].iloc[-1]
def_most_recent_season_median = def_season_median.loc[def_season_median.index != def_final_df['season'].max()].iloc[-1]
for column in model_final_df.columns[4:]:
    model_final_df[column] = model_final_df.apply(fill_missing_values, axis = 1, args = (column, season_median, most_recent_season_median))
for column in def_final_df.columns[4:]:
    def_final_df[column] = def_final_df.apply(fill_missing_values, axis = 1, args = (column, def_season_median, def_most_recent_season_median))

## Merging Offensive Averages with Opponent Defensive Stats
def_final_df = def_final_df.rename(columns={'team_abbr': 'def_team_abbr'})
model_final_df = model_final_df[(model_df['season'] >= 2018)]
model_final_df = pd.merge(
    model_final_df,
    def_final_df,
    left_on=['season', 'week', 'opponent'],
    right_on=['season', 'week', 'def_team_abbr'],
    how='left'
)
model_final_df = model_final_df.drop(columns=['def_team_abbr', 'opponent_y'])
model_final_df = model_final_df.rename(columns={'opponent_x': 'opponent'})

## Model DF Version 4, 5, & 6 Were Created Here
model_final_df.to_csv("data/model_data.csv")