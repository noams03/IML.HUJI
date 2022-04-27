from plotly.subplots import make_subplots

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    df = df[(df.Day >= 1) & (df.Day <= 31)]
    df = df[(df.Month >= 1) & (df.Month <= 12)]
    df = df[(df.Temp >= -30) & (df.Temp <= 55)]
    df['DayOfYear'] = pd.to_datetime(df['Date']).dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_tmp = df[df["Country"] == "Israel"]
    israel_temp_yearly = israel_tmp.groupby('Year')
    fig = make_subplots(1, 4, subplot_titles="")
    for year in israel_temp_yearly.groups:
        fig.add_traces(
            [go.Scatter(x=israel_temp_yearly.get_group(year)["DayOfYear"],
                        y=israel_temp_yearly.get_group(year)[
                            "Temp"],
                        mode='markers', name=year)])
    fig.layout = go.Layout(
        title='Temperature as function of date colored by year',
        xaxis_title="DayOfYear",
        yaxis_title="Temperature",
        height=500)
    fig.show()

    israel_temp_month = israel_tmp.groupby('Month')
    months_std = []
    months_list = list(israel_temp_month.groups)
    for month in months_list:
        months_std.append(np.std(israel_temp_month.get_group(month)["Temp"]))
    fig = go.Figure([go.Bar(x=months_list, y=months_std)],
                    layout=go.Layout(
                        title='std of temerature avg as function of month',
                        xaxis_title="month",
                        yaxis_title="std of the daily temp by month",
                        height=300))
    fig.show()

    # Question 3 - Exploring differences between countries
    country_month_tmp = df.groupby(['Country', 'Month']).Temp.agg(['mean',
                                                    'std']).reset_index()
    fig = px.line(country_month_tmp, x='Month', y='mean',
                  color='Country', error_y='std',
                  title="Avr Temp per month by country ")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    israel_tmp_response = israel_tmp.loc[:, "Temp"]
    israel_dates = israel_tmp.drop(columns="Temp")
    train_x, train_y, test_x, test_y = split_train_test(israel_dates,
                                            israel_tmp_response, 0.75)
    train_x_day_arr = train_x["DayOfYear"].to_numpy()
    train_y_arr = train_y.to_numpy()
    test_x_day_arr = test_x["DayOfYear"].to_numpy()
    test_y_arr = test_y.to_numpy()
    loss_for_k = []
    k_arr = np.linspace(1, 10, 10).astype(int)
    for k in k_arr:
        poly_fit = PolynomialFitting(k)
        poly_fit._fit(train_x_day_arr, train_y_arr)
        loss = round(poly_fit._loss(test_x_day_arr, test_y_arr), 2)
        loss_for_k.append(loss)
        print("loss for " + str(k) + " degree is: " + str(loss))
    fig = go.Figure([go.Bar(x=k_arr, y=loss_for_k)],
                    layout=go.Layout(
                        title='test error recorded for each value of'
                              ' k(polynom degree)',
                        xaxis_title="degree",
                        yaxis_title="test error"))
    fig.show()


    # Question 5 - Evaluating fitted model on different countries
    max_loss = k_arr[loss_for_k.index(min(loss_for_k))]
    train_isr_x_arr = israel_dates["DayOfYear"].to_numpy()
    train_isr_tmp_responses = israel_tmp_response.to_numpy()
    poly_fit = PolynomialFitting(max_loss)
    poly_fit._fit(train_isr_x_arr, train_isr_tmp_responses)
    df_by_country = df.groupby('Country')
    loss_country = []
    country_names = []
    for country in df_by_country.groups:
        if country != "Israel":
            country_tmp_response =\
                df_by_country.get_group(country)["Temp"].to_numpy()
            country_tmp_dates =\
                df_by_country.get_group(country)["DayOfYear"].to_numpy()
            loss_country.append(round(poly_fit._loss(country_tmp_dates,
                                        country_tmp_response), 2))
            country_names.append(country)
    fig = go.Figure([go.Bar(x=country_names, y=loss_country)],
                    layout=go.Layout(
                        title='model’s error over each country based on israel'
                              ' data prediction',
                        xaxis_title="country",
                        yaxis_title="test error"))
    fig.show()
    print(country_names)



