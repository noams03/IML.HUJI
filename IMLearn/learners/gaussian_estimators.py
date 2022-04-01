from __future__ import annotations
import numpy as np
import scipy.stats
from numpy.linalg import inv, det, slogdet


def calculate_normal_density(mu, var, x):
    return 1 / ((2 * np.pi * var) ** 0.5) * np.exp(-0.5 * ((x - mu) /
                                                           np.sqrt(var)) ** 2)

class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased
            estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not
            been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in
            `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in
            `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated
         estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_`
        attribute to `True`
        """
        self.mu_ = np.mean(X)
        sum = 0
        for x in X:
            sum += (x - self.mu_) ** 2
        if self.biased_:
            self.var_ = sum / X.size
        else:
            self.var_ = sum / (X.size - 1)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted
        estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """

        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling "
                             "`pdf` function")
        pdf_arr = np.ndarray(X.size)
        for i, x in enumerate(X):
            pdf_arr[i] = calculate_normal_density(self.mu_, self.var_, x)
        return pdf_arr

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian
        model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        result = 0
        for x in X:
            result += np.log(calculate_normal_density(mu, sigma, x))
        return result


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def calculate_cov(self, X: np.ndarray, first_variate, second_variate):
        sum = 0
        for sample_num in range(len(X)):
            sum += (X[sample_num][first_variate] - self.mu_[first_variate]) \
                   * (X[sample_num][second_variate] - self.mu_[second_variate])
        return sum / (len(X) - 1)

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        mu_arr = np.ndarray(X[0].size)
        for i in range(X[0].size):
            single_variate = X[:, [i]]
            mu_arr[i] = np.mean(single_variate)
        self.mu_ = mu_arr
        cov_mat_dim = X[0].size
        cov_mat = np.ndarray((cov_mat_dim, cov_mat_dim))
        for i in range(cov_mat_dim):
            for j in range(cov_mat_dim):
                cov_mat[i][j] = self.calculate_cov(X, i, j)
        self.cov_ = cov_mat
        self.fitted_ = True
        return self

    def calculate_multivariate_normal_density(mu, cov_mat, x):
        sqrt = np.sqrt(((2*np.pi)**len(x))*np.linalg.det(cov_mat))
        expo = np.exp(-0.5*(x - mu)*np.linalg.inv(cov_mat)*(x - mu))
        return 1/sqrt*expo

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted
        estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        pdf_arr = np.ndarray(X.size)
        for i, x in enumerate(X):
            pdf_arr[i] = self.calculate_multivariate_normal_density\
                (self.mu_, self.cov_, x)
        return pdf_arr


    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray,
                       X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        det_cov = np.linalg.det(cov)
        inv_cuv = np.linalg.inv(cov)
        first_part = ((len(mu) * len(X)) / 2) * np.log(2 * np.pi)
        second_part = (len(X) / 2) * np.log(det_cov)
        third_part = 0
        for i in range(len(X)):
            third_part += np.matmul(np.matmul(X[i] - mu, inv_cuv), X[i] - mu)
        third_part *= 0.5
        return - first_part - second_part - third_part
