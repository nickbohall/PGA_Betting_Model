import pandas as pd
import numpy as np
import datetime as dt

pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 300)

# This file is just for importing the csv Data In. The CSV Data In needs to be updated every year.
# SOURCE: https://www.advancedsportsanalytics.com/pga-raw-data
class DataImport:
    def __init__(self):
        path = "../Data In/2015-2022 raw data tourneylevel condensed.csv"
        self.data = pd.read_csv(path)
        self.data["score"] = self.data.strokes - self.data.hole_par

    def get_data(self):
        return self.data

    def year_list(self):
        return self.data.season.unique()

    def tourney_list(self):
        return self.data["tournament name"].unique()

    def course_list(self):
        return self.data.course.unique()

    def get_raw_data(self):
        self.data = pd.read_csv("../Data In/2015-2022 raw data tourneylevel regression.csv")
        self.data["score"] = self.data.strokes - self.data.hole_par
        return self.data

    # This function is creating a new df at a tournament level with the n_rounds, top score, and winner
    def get_tourn_df(self):
        raw_df = self.get_raw_data()
        df = raw_df.groupby(["tournament name", "date"]).agg({'n_rounds': np.max,
                                                              'score': np.min,
                                                              'course': np.min
                                                            }).reset_index()
        winner_df = raw_df.loc[raw_df.groupby(["tournament name", "date"])['score'].idxmin()]["player"].reset_index(drop=True)
        df = df.merge(winner_df, left_index=True, right_index=True)\
            .rename(columns={'player': 'winner', 'score': 'win_score', 'n_rounds': 'tourn_rounds'})
        return df
