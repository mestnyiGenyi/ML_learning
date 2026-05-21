import numpy as np
import pandas as pd
import random

class MySVM():
    def __init__(self,
                 n_iter = 10,
                 learning_rate = 0.001,
                 weights = None,
                 b = None,
                 C = 1,
                 sgd_sample = None,
                 random_state = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.b = b
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __repr__(self):
        return f"MySVM class: n_iter={self.n_iter!r}, learning_rate={self.learning_rate!r}"   
    
    def predict(self, X):
        X = X.copy()

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        y = np.sign(X @ self.weights + self.b)

        y = np.where(y == -1, 0, y)

        return y.astype(int)


    def get_learning_rate(self, iter):
        if callable(self.learning_rate):
            return self.learning_rate(iter)
        return self.learning_rate   

    def calc_loss(self, y_i, x_i):
        return (self.weights @ self.weights) + self.C * np.mean(max(0, 1 - y_i*(self.weights @ x_i + self.b)))

    def fit(self, X, y, verbose = False):
        random.seed(self.random_state)
        y = y.replace(0, -1)

        n = X.shape[0]
        num_feature = X.shape[1]

        self.weights = np.ones(num_feature)

        self.b = 1

        loss = self.calc_loss(y.loc[0], X.iloc[0])

        if verbose:
            print(f"start | loss: {loss}")

        for i in range(1, self.n_iter + 1):
            if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
            elif isinstance(self.sgd_sample, float):
                sample_rows_idx = random.sample(range(X.shape[0]), round(X.shape[0] * self.sgd_sample))
            else:
                sample_rows_idx = range(X.shape[0]) 
            X_s = X.iloc[sample_rows_idx]
            y_s = y.iloc[sample_rows_idx]
            for j in range(X_s.shape[0]):
                x_i = X_s.iloc[j]
                y_i = y_s.iloc[j]

                if 1 <= y_i*(self.weights @ x_i + self.b):
                    grad_w = 2 * self.weights
                    grad_b = 0
                else:
                    grad_w = 2 * self.weights - self.C * y_i * x_i
                    grad_b = - self.C * y_i
                
                current_lerning_rate = self.get_learning_rate(i)

                self.weights = self.weights - current_lerning_rate * grad_w

                self.b = self.b - current_lerning_rate * grad_b

                loss = self.calc_loss(y_i, x_i)

                if verbose and i % verbose == 0:
                    print(f"{i} | loss: {loss}")

    def get_coef(self) -> tuple:
        return (self.weights.copy(), self.b)
                