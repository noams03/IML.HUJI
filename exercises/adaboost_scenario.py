import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import loss_functions
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    learners_num = np.linspace(1, 250, 250)
    adaboost = AdaBoost(lambda: DecisionStump(), n_learners)
    adaboost.fit(train_X, train_y)
    train_loss = []
    test_loss = []
    for i in range(1, n_learners + 1):
        train_loss.append(adaboost.partial_loss(train_X, train_y, i))
        test_loss.append(adaboost.partial_loss(test_X, test_y, i))
    fig = go.Figure([go.Scatter(x=learners_num, y=train_loss,
                                name="train loss", showlegend=True,
                                mode="markers+lines",
                                line=dict(color="green", width=1))],
                    layout=go.Layout(title=r"$\text{loss of set as function of"
                                           r"number of learners}$",
                                     xaxis={
                                         "title": "Number of learners"},
                                     yaxis={"title": "loss"},
                                     height=400))
    fig.add_trace(go.Scatter(x=learners_num, y=test_loss, name="test loss",
                             mode="markers+lines",
                             line=dict(color="blue", width=1)))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[
                            rf"$\textbf{{{t} iters, loss= {adaboost.partial_loss(test_X, test_y, t)}}}$"
                            for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(
            lambda X: adaboost.partial_predict(X, t), lims[0], lims[1],
            showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                   mode="markers",
                                   showlegend=False,
                                   marker=dict(color=test_y, symbol="circle",
                                               colorscale=[custom[0],
                                                           custom[-1]],
                                               line=dict(color="black",
                                                         width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(
        title=rf"$\textbf{{Decision Boundaries Of Adaboost as function of number of iterations}}$").show()

    # Question 3: Decision surface of best performing ensemble
    fig = go.Figure(
        [decision_surface(lambda X: adaboost.partial_predict(X, 237),
                          lims[0], lims[1],
                          showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(color=test_y, symbol="circle",
                                colorscale=[custom[0],
                                            custom[-1]],
                                line=dict(color="black",
                                          width=1)))])
    fig.update_layout(
        title=rf"$\textbf{{Decision Boundarie by {np.argmin(test_loss) + 1} iterations, the loss:{test_loss[np.argmin(test_loss)]}, accuracy: {loss_functions.accuracy(test_y, adaboost.partial_predict(test_X, np.argmin(test_loss) + 1))}}}$").show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure(
        [decision_surface(lambda X: adaboost.predict(X),
                          lims[0], lims[1],
                          showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(color=train_y, symbol="circle",
                                size=np.abs(
                                    adaboost.D_ / np.max(adaboost.D_) * 5),
                                colorscale=[custom[0],
                                            custom[-1]],
                                line=dict(color="black",
                                          width=1)))])

    fig.update_layout(
        title=rf"$\textbf{{Training set as function of label and weight}}$").show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0, 250)
    fit_and_evaluate_adaboost(0.4, 250)
