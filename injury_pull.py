# Importing Packages
import pandas as pd
import nfl_data_py as nfl

# Naming Year Variable for Data Import (2018 - 2023 Data)
year = 2024
year_list = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# Assigning Data Frames
injury_df= nfl.import_injuries(year_list)

# # Creating CSV Files
# injury_df.to_csv('data/injury_data.csv')