import numpy as np
import pandas as pd
import random

class MyLineReg():
    def __init__(self, 
                n_iter = 100, 
                learning_rate = 0.1, 
                weights = None, 
                metric = None, 
                reg = None, 
                l1_coef = 0, 
                l2_coef = 0,
                sgd_sample = None,
                random_state = 42):
        self.random_state = random_state
        self.sgd_sample = sgd_sample      
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.metric = metric
        self.best_score = 0
        self.weights = weights
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def __repr__(self):
        return f"MyLineReg class: n_iter={self.n_iter!r}, learning_rate={self.learning_rate!r}, weights={self.weights!r}"   


    def calc_metric(self, y, y_pred):
        if self.metric is None:
            return None

        if self.metric == 'mae':
            return np.mean(np.abs(y - y_pred))

        if self.metric == 'mse':
            return np.mean((y - y_pred) ** 2)

        if self.metric == 'rmse':
            return np.sqrt(np.mean((y - y_pred) ** 2))

        if self.metric == 'mape':
            return 100 * np.mean(np.abs((y - y_pred) / y))

        if self.metric == 'r2':
            rss = np.sum((y - y_pred) ** 2)
            tss = np.sum((y - np.mean(y)) ** 2)
            return 1 - rss / tss

    def calc_loss(self, y, y_pred):
        if self.reg is None:
            return np.mean((y_pred - y) ** 2)
        elif self.reg == "l1":
            return (np.mean((y_pred - y) ** 2)) + (self.l1_coef *  np.sum(abs(self.weights)))  
        elif self.reg == "l2":
            return (np.mean((y_pred - y) ** 2)) + (self.l2_coef *  np.sum(self.weights**2))  
        elif self.reg == "elasticnet": 
            return (np.mean((y_pred - y) ** 2)) + (self.l1_coef *  np.sum(abs(self.weights)))  + (self.l2_coef *  np.sum(self.weights**2))  


    def calc_grad(self, y, y_pred, X, sample_rows_idx):
        y_s = y.iloc[sample_rows_idx]
        y_pred_s = y_pred.iloc[sample_rows_idx]
        X_s = X.iloc[sample_rows_idx]
        n = X_s.shape[0]
        if self.reg is None:
            return (2 / n) * ((y_pred_s - y_s) @ X_s)
        elif self.reg == "l1":
            return (2 / n) * ((y_pred_s - y_s) @ X_s) + (self.l1_coef * np.sign(self.weights))
        elif self.reg == "l2":
            return (2 / n) * ((y_pred_s - y_s) @ X_s) + (self.l2_coef * 2 * self.weights)
        elif self.reg == "elasticnet": 
            return (2 / n) * ((y_pred_s - y_s) @ X_s) + (self.l1_coef * np.sign(self.weights)) + (self.l2_coef * 2 * self.weights)


    def get_learning_rate(self, iter):
        if callable(self.learning_rate):
            return self.learning_rate(iter)
        return self.learning_rate    


    def fit(self, X, y, verbose=False):

        random.seed(self.random_state)

        X = X.copy()
        X.insert(0, 'x0', 1)

        n = X.shape[0]
        num_features = X.shape[1]

        self.weights = np.ones(num_features)

        y_pred = X @ self.weights
        loss = self.calc_loss(y, y_pred)

        if verbose:
            if self.metric:
                metric_value = self.calc_metric(y, y_pred)
                print(f"start | loss: {loss} | {self.metric}: {metric_value}")
            else:
                print(f"start | loss: {loss}")

        for i in range(1, self.n_iter + 1):

            if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
            elif isinstance(self.sgd_sample, float):
                sample_rows_idx = random.sample(range(X.shape[0]), round(X.shape[0] * self.sgd_sample))
            else:
                sample_rows_idx = range(X.shape[0])    

            y_pred = X @ self.weights

            grad = self.calc_grad(y,y_pred,X, sample_rows_idx)

            current_lerning_rate = self.get_learning_rate(i)

            self.weights = self.weights - current_lerning_rate * grad

            if verbose and i % verbose == 0:
                y_pred = X @ self.weights
                loss = self.calc_loss(y, y_pred)

                if self.metric:
                    metric_value = self.calc_metric(y, y_pred)
                    print(f"{i} | loss: {loss} | {self.metric}: {metric_value}")
                else:
                    print(f"{i} | loss: {loss}")

        y_pred = X @ self.weights
        self.best_score = self.calc_metric(y, y_pred)
        
    def get_coef(self):
        return self.weights[1:]


    def predict(self, X):
        X = X.copy()
        X.insert(0, "x0", 1)
        return X @ self.weights

    def get_best_score(self):
        return self.best_score







        