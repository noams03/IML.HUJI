from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression, LassoRegression
from sklearn.linear_model import Lasso
from IMLearn.metrics import loss_functions as loss
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    response = lambda x: (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    X = np.linspace(-1.2, 2, n_samples)
    y_src = response(X)
    y = response(X) + np.random.normal(scale=noise, size=n_samples)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y, name=""), 2/3)
    train_X = train_X.to_numpy().reshape(-1)
    train_y = train_y.to_numpy().reshape(-1)
    test_X = test_X.to_numpy().reshape(-1)
    test_y = test_y.to_numpy().reshape(-1)

    fig = go.Figure([go.Scatter(name='real model', x=X, y=y_src, mode='lines')],
              layout=go.Layout(
                  title="train and test samples with noise compared to real"
                        " model",
                  xaxis_title="x",
                  yaxis_title="y"))
    fig.add_traces([
                go.Scatter(name='Train samples', x=train_X, y=train_y, mode="markers",
                           showlegend=False,
                           marker=dict(color='purple'))])
    fig.add_traces([
        go.Scatter(name='Test samples', x=test_X, y=test_y, mode="markers",
                   showlegend=False,
                   marker=dict(color='orange'))])
    fig.show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    deg_arr = np.linspace(0, 10, 11).astype(int)
    train_errors = []
    val_errors = []
    for deg in range(11):
        poly_reg = PolynomialFitting(deg)
        train_err, val_err = cross_validate(poly_reg, train_X, train_y, mean_square_error)
        train_errors.append(train_err)
        val_errors.append(val_err)
    fig = go.Figure(
        [go.Scatter(name='train errors', x=deg_arr, y=train_errors, mode='markers+lines',
                    marker=dict(color='purple')),
         go.Scatter(name='validation errors', x=deg_arr, y=val_errors,
                    mode='markers+lines',
                    marker=dict(color='orange'))],
        layout=go.Layout(
            title="train error and validation error as a function of polynomial"
                  " degree",
            xaxis_title="polynomial degree",
            yaxis_title="error"))

    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(val_errors)
    poly_reg = PolynomialFitting(best_k)
    poly_reg.fit(train_X, train_y)
    test_error = mean_square_error(test_y, poly_reg.predict(test_X))
    print("the value of K* is: ", best_k)
    print("the test error is: %.2f" % test_error)


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    data, target = datasets.load_diabetes(return_X_y=True)
    train_x,  train_y = data[:n_samples], target[:n_samples]
    test_x, test_y = data[n_samples:], target[n_samples:]


    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_values = np.linspace(0, 3, n_evaluations)
    val_error_ridge = []
    train_error_ridge = []
    val_error_lasso = []
    train_error_lasso = []
    for lam in lam_values:
        train_err, val_err = cross_validate(RidgeRegression(lam), train_x, train_y, mean_square_error)
        val_error_ridge.append(val_err)
        train_error_ridge.append(train_err)
        train_err, val_err = cross_validate(Lasso(lam), train_x, train_y,
                                            mean_square_error)
        val_error_lasso.append(val_err)
        train_error_lasso.append(train_err)
    fig = go.Figure(
        [go.Scatter(name='train errors Lasso', x=lam_values, y=train_error_lasso,
                    mode='markers+lines',
                    marker=dict(color='purple')),
         go.Scatter(name='validation errors Lasso', x=lam_values, y=val_error_lasso,
                    mode='markers+lines',
                    marker=dict(color='orange')),
         go.Scatter(name='train errors Ridge', x=lam_values,
                    y=train_error_ridge,
                    mode='markers+lines',
                    marker=dict(color='blue')),
         go.Scatter(name='validation errors Ridge', x=lam_values,
                    y=val_error_ridge,
                    mode='markers+lines',
                    marker=dict(color='green'))],
        layout=go.Layout(
            title="lasso and ridge train and validation error as a function of lambda",
            xaxis_title="lambda",
            yaxis_title="error"))

    fig.show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_lasso = lam_values[np.argmin(val_error_lasso)]
    best_lam_ridge = lam_values[np.argmin(val_error_ridge)]
    print("best lam for lasso: ", best_lam_lasso)
    print("best lam for ridge: ", best_lam_ridge)
    lasso_reg = Lasso(best_lam_lasso)
    rid_reg = RidgeRegression(best_lam_ridge)
    lin_reg = LinearRegression()
    lasso_reg.fit(train_x, train_y)
    rid_reg.fit(train_x, train_y)
    lin_reg.fit(train_x, train_y)
    ridge_error = rid_reg.loss(test_x, test_y)
    lasso_error = loss.mean_square_error(test_y, lasso_reg.predict(test_x))
    linear_error = lin_reg.loss(test_x, test_y)
    print("error for lasso model: ", lasso_error)
    print("error for ridge model: ", ridge_error)
    print("error for linear model: ", linear_error)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()