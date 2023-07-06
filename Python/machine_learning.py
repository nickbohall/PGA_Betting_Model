import pandas as pd
from datetime import datetime as dt
import numpy as np

from player_mapping import PlayerMap
from tourney_data_import import DataImport
from machine_learning_helpers import create_ewma_dynamic, exp_average, linear_regression, logistic_regression

# ------------------------------------ HOUSEKEEPING ------------------------------------#

pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 300)

# ------------------------------------ PLAYER MAP ------------------------------------#

# players = PlayerMap()
# player_list = players.get_player_list()

# ------------------------------------ GET RAW DATA ------------------------------------#

data_in = DataImport()
data = data_in.get_raw_data()
tourn_data = data_in.get_tourn_df()
historical_odds_data = data_in.get_historical_odds()
predict_data = pd.read_csv("../Data Out/2023_The Memorial Tournament pres. by Nationwide_data.csv")

# ------------------------------------ DATA MANIPULATION ------------------------------------#
# Removed "pos" because if they didn't make the cut, it was NA. Didn't know how to handle this well
df = data.loc[:,("player", "date", 'year', 'tournament name', 'course', 'strokes', 'hole_par', "score", "made_cut",
                 "pos", 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total')]
tourn_df = tourn_data
historical_odds_df = historical_odds_data.loc[:, ('player', 'year', 'tournament name', 'odds')]
df = df.merge(tourn_df, on=['tournament name', 'course', 'date'], how="left").merge(
    historical_odds_df, on=['tournament name', 'year', 'player'], how="left")
predict_df = predict_data.loc[:, ('player', 'SG total Average', 'SG PUT Average', 'SG ATG Average', 'SG APR Average', 'SG OTT Average', 'SG TTG Average')]
predict_df.rename(columns={'SG total Average': 'sg_total', 'SG OTT Average': 'sg_ott', 'SG ATG Average': 'sg_arg',
                           'SG TTG Average': 'sg_t2g', 'SG APR Average': 'sg_app', 'SG PUT Average': 'sg_putt'}, inplace=True)

# Adding abs/relative score columns and new finish column to account for cuts
df["abs_score"] = np.where((df.tourn_rounds == 4) & (df.made_cut == 0), df.score * 2, df.score)
df["rel_score"] = df.abs_score - df.win_score
df["new_pos"] = df.groupby(["tournament name", "date"])['abs_score'].rank(ascending=True)

#Removing outliers and bad data
df.drop(df[df.pos > 500].index, inplace=True)

df.sort_values(by=["player", "date"], ascending=[True, True], inplace=True)
df.reset_index(inplace=True)

ml_df = df.loc[:, ("player", "date", 'tournament name', 'course', "score", 'rel_score', "made_cut", 'new_pos',
                   'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total')].dropna()

# Potential Targets
ml_df["top_5"] = np.where(ml_df.new_pos <= 5, 1, 0)
ml_df["top_10"] = np.where(ml_df.new_pos <= 10, 1, 0)

# Dynamic Rolling means
ml_df["sg_total"] = ml_df.groupby("player")["sg_total"].shift().rolling(10, min_periods=2).apply(exp_average, raw=True).values
ml_df["sg_putt"] = ml_df.groupby("player")["sg_putt"].shift().rolling(10, min_periods=2).apply(exp_average, raw=True).values
ml_df["sg_arg"] = ml_df.groupby("player")["sg_arg"].shift().rolling(10, min_periods=2).apply(exp_average, raw=True).values
ml_df["sg_app"] = ml_df.groupby("player")["sg_app"].shift().rolling(10, min_periods=2).apply(exp_average, raw=True).values
ml_df["sg_ott"] = ml_df.groupby("player")["sg_ott"].shift().rolling(10, min_periods=2).apply(exp_average, raw=True).values
ml_df["sg_t2g"] = ml_df.groupby("player")["sg_t2g"].shift().rolling(10, min_periods=2).apply(exp_average, raw=True).values
ml_df["cuts_made"] = ml_df.groupby("player")["made_cut"].shift().rolling(10, min_periods=2).apply(exp_average, raw=True).values

# Course and tourney specific information
ml_df["tourney_avg"] = ml_df.groupby(['player', 'tournament name'])['player', 'rel_score'].shift().rolling(10, min_periods=1).mean().values
ml_df["course_avg"] = ml_df.groupby(['player', 'course'])['player', 'rel_score'].shift().rolling(7, min_periods=1).mean().values

# Dropping columns that we're not using in ML
ml_df.drop(['player', 'date', 'tournament name', 'course'], axis=1, inplace=True)

# ------------------------------------ LINEAR REGRESSION ------------------------------------#
# score_feature_list = ['sg_total', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g',
#                       'tourney_avg', 'course_avg']
# linear_regression(ml_df, target="new_pos", feature_list=score_feature_list,
#                                              title="PGA Linear Regression Model",
#                                              xlabel="Predicted Finish", ylabel="Actual Finish")

# ------------------------------------ LOGISTIC REGRESSION ------------------------------------#
# cut_feature_list = ['sg_total', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g',
#                     'tourney_avg', 'course_avg']
# logistic_regression(ml_df, target="new_pos", feature_list=cut_feature_list,
#                                              title="PGA Logistic Regression Model",
#                                              xlabel="Predicted Cut", ylabel="Actual Cut")
#
# ml_df.to_csv(f"../Data out/ml_test.csv")

# --------------------------------------- PREDICTIONS --------------------------------------- #
pred_df = predict_df.drop("player", axis=1).dropna()
players = predict_df.loc[:, 'player']
score_feature_list = ['sg_total', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g']
prediction = linear_regression(ml_df, target="rel_score", feature_list=score_feature_list, title="PGA Linear Regression Model",
                                             xlabel="Predicted Finish", ylabel="Actual Finish", prediction=True, new_data=pred_df)
test = pd.DataFrame(data = prediction).merge(players, left_index=True, right_index=True)
test.to_csv(f"../Data out/test.csv")