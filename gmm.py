import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters, _compute_precision_cholesky
from sklearn.utils import check_random_state

from parse import args

class CustomGaussianMixture(GaussianMixture):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transform_functions = [np.eye(args.dim) for _ in range(args.proxy_nums)]
        self.lamda = args.lamda
        self.first = False

    def fit(self, X, y=None):
        self.fit_predict(X, y)
        return self

    def fit_predict(self, X, y=None):

        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    log_prob_norm, log_resp = self._e_step(X)
                    self._m_step(X, log_resp)
                    lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                    change = lower_bound - prev_lower_bound
                    self._print_verbose_msg_iter_end(n_iter, change)

                    if abs(change) < self.tol:
                        self.converged_ = True
                        break

                self._print_verbose_msg_init_end(lower_bound)

                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        # if not self.converged_ and self.max_iter > 0:
        #     warnings.warn(
        #         "Initialization %d did not converge. "
        #         "Try different init parameters, "
        #         "or increase max_iter, tol "
        #         "or check for degenerate data." % (init + 1),
        #         ConvergenceWarning,
        #     )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)
    def _e_step(self, X):

        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def _estimate_gaussian_covariances_full(self, resp, X, nk, means, reg_covar, W_k):
        """Estimate the full covariance matrices.

        Parameters
        ----------
        resp : array-like of shape (n_samples, n_components)

        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        means : array-like of shape (n_components, n_features)

        reg_covar : float

        Returns
        -------
        covariances : array, shape (n_components, n_features, n_features)
            The covariance matrix of the current components.
        """
        n_components, n_features = means.shape
        covariances = np.empty((n_components, n_features, n_features))
        for k in range(n_components):
            diff = X - means[k]
            p = np.dot(resp[:, k] * diff.T, diff)
            q = nk[k]

            # covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]

            if self.first:
                covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
            else:
                covariances[k] = (p + W_k[k]) / (q - 4 * self.lamda)

            covariances[k].flat[:: n_features + 1] += reg_covar
        return covariances

    def _estimate_gaussian_covariances_tied(self, resp, X, nk, means, reg_covar):
        """Estimate the tied covariance matrix.

        Parameters
        ----------
        resp : array-like of shape (n_samples, n_components)

        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        means : array-like of shape (n_components, n_features)

        reg_covar : float

        Returns
        -------
        covariance : array, shape (n_features, n_features)
            The tied covariance matrix of the components.
        """
        avg_X2 = np.dot(X.T, X)
        avg_means2 = np.dot(nk * means.T, means)
        covariance = avg_X2 - avg_means2
        covariance /= nk.sum()
        covariance.flat[:: len(covariance) + 1] += reg_covar
        return covariance

    def _estimate_gaussian_covariances_diag(self, resp, X, nk, means, reg_covar):
        """Estimate the diagonal covariance vectors.

        Parameters
        ----------
        responsibilities : array-like of shape (n_samples, n_components)

        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        means : array-like of shape (n_components, n_features)

        reg_covar : float

        Returns
        -------
        covariances : array, shape (n_components, n_features)
            The covariance vector of the current components.
        """
        avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
        avg_means2 = means ** 2
        avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
        return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar

    def _estimate_gaussian_covariances_spherical(self, resp, X, nk, means, reg_covar):
        """Estimate the spherical variance values.

        Parameters
        ----------
        responsibilities : array-like of shape (n_samples, n_components)

        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        means : array-like of shape (n_components, n_features)

        reg_covar : float

        Returns
        -------
        variances : array, shape (n_components,)
            The variance values of each components.
        """
        return self._estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(1)

    def _estimate_gaussian_parameters(self, X, resp, reg_covar, covariance_type, W_k):
        """Estimate the Gaussian distribution parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data array.

        resp : array-like of shape (n_samples, n_components)
            The responsibilities for each data sample in X.

        reg_covar : float
            The regularization added to the diagonal of the covariance matrices.

        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.

        Returns
        -------
        nk : array-like of shape (n_components,)
            The numbers of data samples in the current components.

        means : array-like of shape (n_components, n_features)
            The centers of the current components.

        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.
        """
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        covariances = {
            "full": self._estimate_gaussian_covariances_full,
            "tied": self._estimate_gaussian_covariances_tied,
            "diag": self._estimate_gaussian_covariances_diag,
            "spherical": self._estimate_gaussian_covariances_spherical,
        }[covariance_type](resp, X, nk, means, reg_covar, W_k)
        return nk, means, covariances

    def normalize_and_exp(self, vector):
        # L2 归一化
        normalized_vector = vector / (np.linalg.norm(vector) + self.reg_covar)
        # 取绝对值
        normalized_vector = np.abs(normalized_vector) - 0.5
        # 指数函数处理
        exp_vector = np.exp(normalized_vector)

        return exp_vector

    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        functions = self.transform_functions
        D_k = []
        for i in range(self.n_components):
            result_matrix = np.matmul(self.means_[i], functions[i])
            o_k = (self.means_[i] - result_matrix)
            s_k = self.normalize_and_exp(o_k)
            square_matrix = np.eye(len(s_k)) * s_k + (1 - np.eye(len(s_k)))

            w_k = -4 * self.lamda * (self.covariances_[i] * square_matrix)
            D_k.append(w_k)

        self.weights_, self.means_, self.covariances_ = self._estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type, D_k
        )

        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, self.covariance_type)
