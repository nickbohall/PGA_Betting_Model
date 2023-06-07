import pandas as pd
import numpy as np
from tourney_data_import import DataImport

class TourneyFinishes:
    def __init__(self):
        self.data = DataImport().get_data()

    def average_score_by_tourney(self, tourney):
        try:
            tourney_df = self.data.loc[(self.data["tournament name"] == tourney)]

            score_df = tourney_df[["player", "score", "made_cut"]]
            score_df = score_df.rename(columns={"score": "avg_score_tourney", "made_cut": "avg_cut_tourney"})

            average_score = score_df.groupby("player").mean().round(2).sort_values(by= "avg_score_tourney")
            average_score = average_score.reset_index(drop=False)

            print("tourney scores df successfully gathered")
            return average_score
        except:
            print("Failed to find that tournament in the data")
            return np.nan

    def average_score_by_course(self, course):
        try:
            tourney_df = self.data.loc[(self.data.course == course)]

            score_df = tourney_df[["player", "score", "made_cut"]]
            score_df = score_df.rename(columns={"score": "avg_score_course", "made_cut": "avg_cut_course"})

            average_score = score_df.groupby("player").mean().round(2).sort_values(by= "avg_score_course")
            average_score = average_score.reset_index(drop=False)

            print("course scores df successfully gathered")
            return average_score
        except:
            print("failed to find that course in the data")
            return np.nan

    def last_season_score(self, tourney, season):
        try:
            last_season = int(season) - 1
            tourney_df = self.data.loc[(self.data["tournament name"] == tourney) & (self.data["season"] == last_season)]

            score_df = tourney_df[["player", "score", "made_cut"]]
            score_df = score_df.rename(columns={"score": f"{last_season}_score", "made_cut": f"{last_season}_cut"})

            last_score = score_df.reset_index(drop=False)

            print("last season tourney df successfully gathered")
            return last_score
        except:
            print("Can't find a score for last season")
            return np.nan

