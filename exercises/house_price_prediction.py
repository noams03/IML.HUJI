from IMLearn.learners.regressors import LinearRegression
from IMLearn.utils.utils import *
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()

    df = df[df["price"] > 0]
    df = df[df["bedrooms"] > 0]
    df = df[df["bathrooms"] > 0]
    df = df[df["sqft_living"] >= 200]
    df = df[df["sqft_lot"] > 520]
    df = df[df["floors"] > 0]
    df = df[(df.waterfront == 1) | (df.waterfront == 0)]
    df = df[(df.view >= 0) & (df.view <= 4)]
    df = df[(df.condition >= 1) & (df.condition <= 5)]
    df = df[(df.grade >= 1) & (df.grade <= 13)]
    df = df[df.sqft_above > 290]
    df = df[df.yr_built >= 0]
    df = df[df.yr_renovated >= 0]
    df = df[df.lat > 47]
    df = df[(df.long >= -123) & (df.long <= -120)]
    df = df[df.sqft_living15 >= 399]
    df = df[df.sqft_lot15 >= 651]

    y = df.loc[:, 'price']
    X = df.loc[:,
        ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
         "waterfront", "view", "condition", "grade", "sqft_above",
         "sqft_basement", "yr_built", "yr_renovated", "lat",
         "long", "sqft_living15", "sqft_lot15"]]
    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    pearson_vec = np.ndarray(len(X.columns))
    price_std = np.std(y)
    for i in range(len(X.columns)):
        feature_vec = X.iloc[:, i]
        feature_std = np.std(feature_vec)
        cov = np.cov(feature_vec, y)
        pearson_vec[i] = cov[0][1] / (price_std * feature_std)
        go.Figure([go.Scatter(x=X.iloc[:, i], y=y, mode='markers')],
                  layout=go.Layout(
                      title=('correlation between price and ' + str(
                          X.columns[i]) +
                             ' is: ' + str(pearson_vec[i])),
                      xaxis_title="feature values",
                      yaxis_title="responses",
                      height=300)).write_image(str(output_path) + "/price_" +
                                               str(X.columns[
                                                    i]) + "_correlation.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)


    # Question 4 - Fit model over increasing percentages of the overall
    # training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10
    # times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error
    # ribbon of size (mean-2*std, mean+2*std)
    train_X['price'] = train_y
    test_X_array = test_X.to_numpy()
    test_y_array = test_y.to_numpy()
    average_loss_vec = np.ndarray(91)
    std_vec = np.ndarray(91)
    for p in range(10, 101):
        mean_loss_p =[]
        for i in range(10):
            frac_train_set = train_X.sample(frac=p/100)
            frac_response_vec = frac_train_set.loc[:, 'price']
            frac_train_set = frac_train_set.drop(columns='price')
            frac_train_set_array = frac_train_set.to_numpy()
            frac_response_vec_array = frac_response_vec.to_numpy()
            l = LinearRegression()
            l.fit(frac_train_set_array, frac_response_vec_array)
            mean_loss_p.append(l.loss(test_X_array, test_y_array))
        average_loss_vec[(p-10)] = (np.mean(mean_loss_p))
        std_vec[(p - 10)] = (np.std(mean_loss_p))
    p_array = np.linspace(10, 100, 91)
    fig = go.Figure(data=(
            (go.Scatter(x=p_array, y=average_loss_vec, mode='markers+lines',
                        name= 'mean loss per training size')),
             (go.Scatter(x=p_array, y=average_loss_vec - 2 * std_vec,
               fill=None,
               mode="lines", line=dict(color="lightgrey"),
               showlegend=False)),
             (go.Scatter(x=p_array, y=average_loss_vec + 2 * std_vec,
               fill='tonexty',
               mode="lines", line=dict(color="lightgrey"),
               showlegend=False))),
        layout=go.Layout(
            title=("Mean loss and confidence interval as function of "
                   "training set"),
            xaxis_title="training size in percentage",
            yaxis_title="mean loss",
            height=500))
    fig.show()
