from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1
    X = np.random.normal(mu, var, 1000)
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(X)
    print(univariate_gaussian.mu_, univariate_gaussian.var_)

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    estimated_mean = []
    for m in ms:
        X_samples = X[0:m]
        estimated_mean.append(np.abs(univariate_gaussian.fit(X_samples).mu_ -
                                     mu))
    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{Distance between Estimation of "
                        r"Expectation to "
                        r"real Expectation As Function Of Number Of Samples}$",
                  xaxis_title="$\text{ number of samples}$",
                  yaxis_title="Dist",
                  height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_arr = univariate_gaussian.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdf_arr, mode='markers',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"Density As Function of sample ",
                  xaxis_title="$\\text{  samples}$",
                  yaxis_title="Pdf",
                  height=400)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])
    cov_mat = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0],
                        [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mean, cov_mat, 1000)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(X)
    print("mu:")
    print(multivariate_gaussian.mu_)
    print("var:")
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f1_f3_arr = np.linspace(-10, 10, 200).astype(float)
    likelihood_mat = np.ndarray((200, 200))
    for row, i in enumerate(f1_f3_arr):
        for col, j in enumerate(f1_f3_arr):
            likelihood_mat[row][col] = multivariate_gaussian.log_likelihood \
                (np.array([i, 0, j, 0]), cov_mat, X)
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(x=f1_f3_arr, y=f1_f3_arr, z=likelihood_mat,
                   colorbar=dict(title="Log likelihood")))
    fig.update_layout(
        title="Log likelihood of samples with change expectation in "
              "two variates",
        xaxis_title="Values of f3",
        yaxis_title="Values of f1")
    fig.show()

    # Question 6 - Maximum likelihood
    f1, f3 = np.unravel_index(likelihood_mat.argmax(), likelihood_mat.shape)
    print("values of f1 and f3 achieved the maximum log-likelihood:")
    print("f1:", f1_f3_arr[f1])
    print("f3:", f1_f3_arr[f3])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
