# This is all based off of Tim Bryan's 2022 YouTube video explaining how to pull data from the NFL's Next Gen Stats
# The link to view that video is: https://www.youtube.com/watch?v=wWgGgmqijNU

# Importing Packages
import pandas as pd
import nfl_data_py as nfl

pd.set_option('display.max_columns', None)

# Naming Year Variable for Data Import (2016 - 2023 Data)
year = 2024
year_list = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# Assigning Data Frames
df_passing_uncl = nfl.import_ngs_data(stat_type = "passing")
df_receiving_uncl = nfl.import_ngs_data(stat_type = "receiving")
df_rushing_uncl = nfl.import_ngs_data(stat_type = "rushing")
df_wins_uncl = nfl.import_win_totals(year_list)


# Cleaning Data Frames Using NFL Package
df_passing = nfl.clean_nfl_data(df_passing_uncl)
df_receiving = nfl.clean_nfl_data(df_receiving_uncl)
df_rushing = nfl.clean_nfl_data(df_rushing_uncl)
df_wins = nfl.clean_nfl_data(df_wins_uncl)

# # Creating CSV Files
# df_passing.to_csv('data/passing_data.csv')
# df_receiving.to_csv('data/receiving_data.csv')
# df_rushing.to_csv('data/rushing_data.csv')
# df_wins.to_csv('data/weekly_ml_odds.csv')