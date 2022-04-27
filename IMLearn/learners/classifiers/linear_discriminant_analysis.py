from typing import NoReturn

import scipy

from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import loss_functions


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = \
            None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector,
        same covariance matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        classes, times = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.pi_ = times / len(y)
        mu_arr = np.zeros((len(classes), len(X[0])))
        cov_mat = np.zeros((len(X[0]), len(X[0])))
        for i, cls in enumerate(self.classes_):
            X_class = X[y == cls]
            mu_arr[i] = np.mean(X_class, axis=0)
        self.mu_ = mu_arr
        for i in range(len(X)):
            var_mat = (X[i] - self.mu_[np.where(classes == y[i])])
            cov_mat += (var_mat.T @ var_mat)
        self.cov_ = cov_mat / (len(X) - len(classes))
        self._cov_inv = np.linalg.inv(self.cov_)
        self.fitted_ = True



    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` "
                "function")
        like = np.empty((len(X), len(self.classes_)))
        for i in range(len(X)):
            for k in range(len(self.classes_)):
                like[i][k] = (self._cov_inv @ self.mu_[k].T).T @ X[i].T
                like[i][k] += np.log(self.pi_[k]) - (0.5 * (self.mu_[k] @
                                                                self._cov_inv @
                                                                self.mu_[k].T))
        return like

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
        return loss_functions.misclassification_error(y, y_pred)
