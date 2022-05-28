from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes, \
    linear_discriminant_analysis, gaussian_naive_bayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from math import atan2, pi


pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    def callback_func(fit: Perceptron, x: np.ndarray, y: int):
        losses.append(fit._loss(X, y_true))

    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y_true = load_dataset(r"../datasets/" + f)
        # Fit Perceptron and record loss in each fit iteration
        losses = []
        per = Perceptron(callback=callback_func)
        per.fit(X, y_true)

        # Plot figure
        iter_arr = np.linspace(1, per.max_iter_, per.max_iter_)
        go.Figure([go.Scatter(x=iter_arr, y=losses, mode='markers+lines',
                              name=r'$\widehat\mu$')],
                  layout=go.Layout(
                      title="loss as function of iteration num in perceptron"
                            " with " + n + " data",
                      xaxis_title="Iteration number",
                      yaxis_title="Loss",
                      height=600)).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1
     and gaussians2 datasets
    """
    lda = LDA()
    qda = GaussianNaiveBayes()
    models = np.array(["lda", "qda"])
    symbols = np.array(["circle", "x"])
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y_true = load_dataset(r"../datasets/" + f)
        # Fit models and predict over training set
        lda.fit(X, y_true)
        y_pred_lda = lda._predict(X)
        qda.fit(X, y_true)
        y_pred_qda = qda._predict(X)
        y_pred = np.array([y_pred_lda, y_pred_qda]).astype(int)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        err_lda = [1 if y_true[i] == y_pred_lda[i] else 0 for i in
                   range(len(y_true))]
        err_na = [1 if y_true[i] == y_pred_qda[i] else 0 for i in
                  range(len(y_true))]

        loss_vecs = np.array([err_lda, err_na])

        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            rf"{{Model: {models[i]} , Accuracy: {accuracy(y_true, y_pred[i])}"
            for i in range(len(y_pred))],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        for i, m in enumerate(y_pred):
            fig.add_traces([
                go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                           showlegend=False,
                           marker=dict(color=y_pred[i],
                                       symbol=symbols[loss_vecs[i]],
                                       line=dict(color="black", width=1)))],
                rows=(i // 3) + 1, cols=(i % 3) + 1)

            fig.update_layout(
                title=rf"$\textbf{{{f} Dataset, compare between LDA and QDA}}$",
                margin=dict(t=100)) \
                .update_xaxes(visible=False).update_yaxes(visible=False)

            for c in range(len(lda.classes_)):
                fig.add_traces(get_ellipse(lda.mu_[c], lda.cov_),
                               rows=1, cols=1)

            for m in range(len(qda.classes_)):
                fig.add_traces(
                    get_ellipse(qda.mu_[m], np.diag(qda.vars_[m])),
                    rows=1, cols=2)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

