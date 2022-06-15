from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    kfolds_x = np.array_split(X, cv)
    kfolds_y = np.array_split(y, cv)
    errors_folds_train = 0
    errors_folds_val = 0
    for i in range(cv):
        x_k = np.concatenate((*kfolds_x[:i], *kfolds_x[i+1:]))
        y_k = np.concatenate((*kfolds_y[:i], *kfolds_y[i+1:]))
        estimator.fit(x_k, y_k)
        errors_folds_train += scoring(estimator.predict(x_k), y_k)
        errors_folds_val += scoring(estimator.predict(kfolds_x[i]), kfolds_y[i])
    return errors_folds_train/cv, errors_folds_val/cv
