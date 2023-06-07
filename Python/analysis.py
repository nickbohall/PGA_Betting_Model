import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class Analysis:
    def get_analysis_df(self, data, season, tourney):
        try:
            data_clean = data.drop(columns=["index"], axis=1)

            data_clean["score"] = -data_clean.iloc[:, 4:9].sum(axis=1) + data_clean["scoring Average"]
            data_clean["tourney_rank"] = data_clean.avg_score_tourney.rank()
            data_clean["course_rank"] = data_clean.avg_score_tourney.rank()

            small_df = data_clean[["player", "score", "odds", 'tourney_rank', 'course_rank']].sort_values("score", ascending=True)

            small_df.to_csv(f"../Data out/{season}_{tourney}_analysis.csv")

        except:
            print("Analysis step failed somewhere.")

