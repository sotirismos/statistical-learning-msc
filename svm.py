import numpy as np
from cvxopt import matrix, solvers


class MoschosSVM:
    def __init__(self, kernel: str):
        self.kernel = kernel
        self.kernel_function = None
        self.sv_x = None
        self.sv_y = None
        self.alphas = None
        self.b = 0
        self.coef0 = 0.0

    def set_params(self, **params):
        if "C" in params:
            self.C = params["C"]
        if "degree" in params:
            self.degree = params["degree"]
        if "gamma" in params:
            self.gamma = params["gamma"]
        if "coef0" in params:
            self.coef0 = params["coef0"]

    def _linear_kernel(self, x1, x2):
        """
        Compute the linear kernel between two vectors.

        Parameters:
            x1 (np.ndarray): First input vector.
            x2 (np.ndarray): Second input vector.

        Returns:
            float: Linear kernel result.
        """
        return np.dot(x1, x2)

    def _polynomial_kernel(self, x1, x2):
        """
        Compute the polynomial kernel between two vectors.

        Parameters:
            x1 (np.ndarray): First input vector.
            x2 (np.ndarray): Second input vector.
            degree (int): Degree of the polynomial.
            coef0 (float): Independent term in kernel function.

        Returns:
            float: Polynomial kernel result.
        """
        return (np.dot(x1, x2) + self.coef0) ** self.degree

    def _rbf_kernel(self, x1, x2):
        """
        Compute the Radial Basis Function (RBF) kernel between two vectors.

        Parameters:
            x1 (np.ndarray): First input vector.
            x2 (np.ndarray): Second input vector.
            gamma (float): Kernel coefficient.

        Returns:
            float: RBF kernel result.
        """
        distance = np.linalg.norm(x1 - x2)
        return np.exp(-self.gamma * distance**2)

    def _mlp_kernel(self, x1, x2):
        """
        Compute the sigmoid (MLP) kernel between two vectors.

        Parameters:
            x1 (np.ndarray): First input vector.
            x2 (np.ndarray): Second input vector.
            gamma (float): Slope parameter in sigmoid function.
            coef0 (float): Intercept parameter in sigmoid function.

        Returns:
            float: Sigmoid kernel result.
        """
        return np.tanh(self.gamma * np.dot(x1, x2) + self.coef0)

    def _compute_kernel_matrix(self, X):
        """
        Compute the kernel matrix K for the dataset X using the specified kernel function.

        Parameters:
            X (np.ndarray): Input data matrix of shape (n_samples, n_features).
            kernel (callable): Kernel function.

        Returns:
            np.ndarray: Kernel matrix of shape (n_samples, n_samples).
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if self.kernel_function is not None:
                    K[i, j] = self.kernel_function(X[i], X[j])
                else:
                    raise ValueError("No kernel function set.")
        return K

    def fit(self, X_train, y_train):
        """
        Train an SVM classifier using a custom kernel and predict labels for test data.

        Parameters:
            X_train (np.ndarray): Training data features of shape (n_samples, n_features).
            y_train (np.ndarray): Training data labels of shape (n_samples,).

        Returns:
            np.ndarray: Predicted labels for the test data.
        """
        n_samples, _ = X_train.shape

        # Select the kernel function
        if self.kernel == "linear":
            self.kernel_function = self._linear_kernel
        elif self.kernel == "poly":
            self.kernel_function = self._polynomial_kernel
        elif self.kernel == "rbf":
            self.kernel_function = self._rbf_kernel
        elif self.kernel == "mlp":
            self.kernel_function = self._mlp_kernel
        else:
            raise ValueError(
                f"Unsupported kernel type '{self.kernel}'. Supported kernels are 'linear', 'polynomial', 'rbf', and 'mlp'."
            )

        # Compute the kernel matrix
        K = self._compute_kernel_matrix(X_train)

        # Set up parameters for quadratic programming
        y_train = y_train.astype(float)
        P = matrix(np.outer(y_train, y_train) * K)
        q = matrix(-np.ones(n_samples))
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = matrix(y_train.reshape(1, -1))
        b = matrix(np.zeros(1))

        # Solve the quadratic programming problem
        solvers.options["show_progress"] = False
        solution = solvers.qp(P, q, G, h, A, b)

        # Extract Lagrange multipliers
        alphas = np.ravel(solution["x"])

        # Support vectors have non-zero Lagrange multipliers
        tol = 1e-5

        i = np.where(alphas > tol)[0]

        self.alphas = alphas[i]
        self.sv_X = X_train[i]
        self.sv_y = y_train[i]

        # Compute the bias term
        self.b = np.mean(
            [
                self.sv_y[i]
                - np.sum(
                    self.alphas
                    * self.sv_y
                    * np.array(
                        [self.kernel_function(self.sv_X[i], x_j) for x_j in self.sv_X]
                    )
                )
                for i in range(len(self.alphas))
            ]
        )

    def pred(self, X):
        """
        Compute the decision function for input samples X.

        Parameters:
            X (np.ndarray): Input data of shape (m_samples, n_features).

        Returns:
            np.ndarray: Decision function values for each input sample.
        """
        result = np.zeros(X.shape[0])
        if self.alphas is not None:
            for i in range(len(self.alphas)):
                if self.kernel_function is not None:
                    K_sv = np.array([self.kernel_function(self.sv_X[i], x) for x in X])
                else:
                    raise ValueError("No kernel function set.")
                # Supressing Pylance for the following line (don't judge me)
                result += self.alphas[i] * self.sv_y[i] * K_sv  # type: ignore

            predictions = result + self.b
            return np.sign(predictions)
