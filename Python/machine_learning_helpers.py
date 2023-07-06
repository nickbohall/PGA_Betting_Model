# Basic Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Visualizations
import seaborn as sns
import statsmodels.api as sm
# Modeling
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def dynamic_window_ewma(x):
    # Calculate rolling exponentially weighted EPA with a dynamic window size
    values = np.zeros(len(x))

    for i, (_, row) in enumerate(x.iterrows()):
        sg = x.sg_shifted[:i + 1]

        values[i] = sg.ewm(min_periods=1, span=30).mean().values[-1]

    return pd.Series(values, index=x.index)


def create_sg_df(data, sg_type, side):
    # Return a df based on the play data, sg type, and offense vs defense
    return data.loc[data[sg_type] == 1, :].groupby([side, 'season', 'date'], as_index=False)['sg'].mean()


def lag_df(df, side):
    return df.groupby(side)['sg'].shift()


def create_ewma(df, side):
    return df.groupby(side)['sg_shifted'].transform(lambda x: x.ewm(min_periods=1, span=10).mean())


def create_ewma_dynamic(df, side):
    return df.groupby(side).apply(dynamic_window_ewma).values


# Kernel function ( backward-looking exponential )
def K(x):
    return np.exp(-np.abs(x)) * np.where(x <= 0, 1, 0)


# Exponenatial average function
def exp_average(values):
    N = len(values)
    exp_weights = list(map(K, np.arange(-N, 0) / N))
    return values.dot(exp_weights) / N


def graph_linear_model(y_pred, y_test, r2="NA", mse="NA", title="NA", xlabel="NA", ylabel="NA"):
    # set x and y
    x = y_pred
    y = y_test

    # calculate equation for trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    # Create the subplot function
    fig, ax = plt.subplots()

    # Plot the scatter
    ax.scatter(x, y)

    # Plot the trendline
    ax.plot(x, p(x), color="red")

    # Titles and axes
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Plotting some text
    plt.annotate('R-squared = %0.4f' % r2, xy=(0.025, 0.95), xycoords='axes fraction')
    plt.annotate("y = %.2fx + %.2f" % (z[0], z[1]), xy=(0.025, 0.90), xycoords='axes fraction')
    plt.annotate(f"MSE = {mse: .3f}", xy=(0.025, 0.85), xycoords='axes fraction')

    plt.show()


def linear_regression(df, target, feature_list, title="", xlabel="", ylabel="", prediction=False, new_data=""):
    # Dropping NA's and the 2020 season because covid and testing
    df_l = df.dropna()
    print(df_l.isna().values.any())

    # Defining features & target
    x = df_l[target]
    y = df_l[feature_list]

    # Creating train test split. Test will be 20% of the data
    X_train, X_test, y_train, y_test = train_test_split(y, x, test_size=0.2, random_state=10)

    # Creating a spot to test a new dataset if I want after I train and test
    if prediction == True:
        X_test = new_data

    # Create Regression object and fit
    clf = LinearRegression()
    clf.fit(X_train, y_train)

    # Make a prediction based on the fit. Use X_test (20% of the data)
    y_pred = clf.predict(X_test)

    if prediction == True:
        return y_pred

    # Get some info about the coefficients and how they're impacting the model
    mod = sm.OLS(y_train, X_train)
    fii = mod.fit()

    # Getting some info for graphing later
    r2 = fii.rsquared
    mse = mean_squared_error(y_pred, y_test)

    summary = fii.summary()
    print(summary)

    # Call graphing function to output graph
    graph_linear_model(y_pred, y_test, r2, mse, title, xlabel, ylabel)

    return summary


def logistic_regression(df, target, feature_list, title="", xlabel="", ylabel=""):
    # Dropping NA's and the 2020 season because covid and testing
    df_l = df.dropna()

    # Defining features & target
    x = df_l[target]
    y = df_l[feature_list]

    # Creating train test split. Test will be 20% of the data
    X_train, X_test, y_train, y_test = train_test_split(y, x, test_size=0.2, random_state=10)

    # Create Regression object and fit
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Make a prediction based on the fit. Use X_test (20% of the data)
    y_pred = clf.predict(X_test)

    # Get some info about the coefficients and how they're impacting the model
    mod = sm.Logit(y_train, X_train)
    fii = mod.fit()

    # Getting some info for graphing later
    mse = mean_squared_error(y_pred, y_test)

    summary = fii.summary()
    print(summary)
    # Plotting - we don't need a separate function for this one
    sns.regplot(x=y_pred, y=y_test, data=df, logistic=True)
    plt.show()

    # Call make predicitons function
    # predictions = make_predictions(df, clf, features=feature_list)

    return summary


