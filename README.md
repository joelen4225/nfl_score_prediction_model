# 2024-25 NFL Score Prediction Model
This is all the data, scripts, and outputs I have for my new score prediction model that showcases a Voting Regressor. From 2021 - 2023, I used a basic team rating approach on Google Sheets to predict the outcomes of NFL games. I wanted to switch to a modeling approach to make the process of score prediction more detailed and to challenge myself in Python.

Included in the repository is multiple scripts that highlight where I am pulling data from as well as the process I use to prep the data for modeling. You can see that missing values were taken care of by using team-by-team season averages. The 'data' folder and the 'view_df.ipynb' notebook highlight data that was used in both notebook output format and CSV format. If there is any data missing or something you would like to clear up, please reach out to the email below!

My modeling technique is a Voting Regressor of a Random Forest Regression and LightGBM model. I then scale the features and choose the 205 most important ones (this number was chosen through trial and error of modeling). In the 'modeling_attempts' folder, you can find all the different ways I tried to predict the scores before choosing my final attempt. In the 'outcomes' folder, you will find two versions of finalized data for the 2023 season after running my model through it. It is key to note the 'outcomes' folder was last updated 8/14/2024. For a more updated version, please reach out to the email below!

It is important to note that for my own personal use, I have included the 2023 data within the training data and will be using the upcoming 2024 games as my new "test" set. I expect early weeks to perform poorly compared to later ones due to the lack of curren-season data presence, causing my average batching technique to supply many duplicate datapoints. 

Thank you for taking a look at my project and I hope you enjoy! Any questions or advice would be greatly appreciated. You can reach me at joseph.leonard725@gmail.com. 

