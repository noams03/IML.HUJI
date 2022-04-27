from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import loss_functions


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        classes, times = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.pi_ = times/len(y)
        mu_arr = np.zeros((len(classes), len(X[0])))
        var_mat = np.zeros((len(classes), len(X[0])))
        for i, cls in enumerate(self.classes_):
            X_class = X[y == cls]
            mu_arr[i] = np.mean(X_class, axis=0)
            var_mat[i] = np.var(X_class, axis=0)
        self.mu_ = mu_arr
        self.vars_ = var_mat

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

    def pdf_calc(self, X: np.ndarray, mu, var) -> np.ndarray:
        return 1 / ((2 * np.pi * var) ** 0.5) * np.exp(-0.5 * ((X - mu) /
                                                           np.sqrt(var)) ** 2)

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
            raise ValueError("Estimator must first be fitted before calling"
                             " `likelihood` function")
        like = np.empty((len(X), len(self.classes_)))
        for k in range(len(self.classes_)):
            dis_class = self.pdf_calc(X, self.mu_[k], self.vars_[k]) *self.pi_[k]
            like[:, k] = np.prod(dis_class, axis=1)
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
