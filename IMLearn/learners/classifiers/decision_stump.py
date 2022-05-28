from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np

from itertools import product

from ...metrics import loss_functions


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        select_feature = 0
        thr = 0
        thr_err = 1
        sign = 0
        for j in range(X.shape[1]):
            temp_thr_plus, temp_err_plus = self._find_threshold(X[:, j], y, 1)
            temp_thr_minus, temp_err_minus = self._find_threshold(X[:, j], y, -1)
            if temp_err_plus < thr_err:
                thr_err = temp_err_plus
                thr = temp_thr_plus
                sign = 1
                select_feature = j
            if temp_err_minus < thr_err:
                thr_err = temp_err_minus
                thr = temp_thr_minus
                sign = -1
                select_feature = j
        self.threshold_ = thr
        self.j_ = select_feature
        self.sign_ = sign
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)


    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sort_ind = np.argsort(values)
        sorted_values, sorted_labels = values[sort_ind], labels[sort_ind]
        thr = [(sorted_values[i] + sorted_values[i + 1]) / 2 for i in
                  range(len(values) - 1)]
        thr = np.concatenate([[-np.inf], thr, [np.inf]])
        min_loss = np.sum(abs(sorted_labels[np.sign(sorted_labels) == sign]))
        losses = np.append(min_loss,
                           min_loss - np.cumsum((sorted_labels * sign)))
        ind = np.argmin(losses)
        return thr[ind], losses[ind]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        return np.sum(np.where(np.sign(y) != y_pred, np.abs(y), 0))
