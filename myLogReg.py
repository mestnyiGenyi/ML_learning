import numpy as np
import pandas as pd
import random

class MyLogReg():
    def __init__(self, 
                 n_iter = 10, 
                 learning_rate = 0.1,
                 weights = None,
                 reg = None, 
                 l1_coef = 0, 
                 l2_coef = 0,
                 metric = None,                 
                 sgd_sample = None,
                 random_state = 42):
        self.random_state = random_state
        self.sgd_sample = sgd_sample                       
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef                 
        self.best_score = 0
        self.metric = metric
        self.weights = weights         
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def get_confusion_matrix(self, y, y_pred):
        y = np.array(y)
        y_pred = np.array((y_pred > 0.5).astype(int))

        tp = np.sum((y == 1) & (y_pred == 1))
        fp = np.sum((y == 0) & (y_pred == 1))
        fn = np.sum((y == 1) & (y_pred == 0))
        tn = np.sum((y == 0) & (y_pred == 0))

        return np.array([[tp, fn],
                        [fp, tn]])


    def __repr__(self):
        return f"MyLogReg class: n_iter={self.n_iter!r}, learning_rate={self.learning_rate!r}" 

    def calc_metric(self, y, y_pred):
        
        mc = self.get_confusion_matrix(y, y_pred)

        if self.metric is None:
            return None

        if self.metric == 'accuracy':
            return (mc[0][0] + mc[1][1])/ np.sum(mc)

        if self.metric == 'precision':
            return mc[0][0] / (mc[0][0] + mc[1][0])

        if self.metric == 'recall':
            return mc[0][0] / (mc[0][0] + mc[0][1])

        if self.metric == 'f1':
            return 2 * (mc[0][0] / (mc[0][0] + mc[1][0])) * (mc[0][0] / (mc[0][0] + mc[0][1])) / ((mc[0][0] / (mc[0][0] + mc[0][1])) +  (mc[0][0] / (mc[0][0] + mc[1][0])))

        if self.metric == 'roc_auc':
            y = np.array(y)
            y_pred = np.array(y_pred)
            y_pred = np.round(y_pred, 10)

            y_pos = y_pred[y==1]
            y_neg = y_pred[y==0]

            score_sum = 0

            for y_p in y_pos:
                for y_n in y_neg:
                    if y_p > y_n:
                        score_sum += 1
                    elif y_p == y_n:
                        score_sum += 0.5

            return (1 / (len(y_pos) * len(y_neg))) * score_sum

    def calc_loss(self, y, y_pred):
        loss = (-np.mean(y * np.log(y_pred) + (1 - y) * (np.log(1 - y_pred))))

        if self.reg is None:
            return loss
        elif self.reg == "l1":
            return loss + (self.l1_coef *  np.sum(abs(self.weights)))  
        elif self.reg == "l2":
            return loss + (self.l2_coef *  np.sum(self.weights**2))  
        elif self.reg == "elasticnet": 
            return loss + (self.l1_coef *  np.sum(abs(self.weights)))  + (self.l2_coef *  np.sum(self.weights**2))  


    def calc_grad(self, y, y_pred, X, sample_rows_idx):
        y_s = y.iloc[sample_rows_idx]
        y_pred_s = y_pred.iloc[sample_rows_idx]
        X_s = X.iloc[sample_rows_idx]
        n = X_s.shape[0]

        grad =  (1 / n) * ((y_pred_s - y_s) @ X_s) 

        if self.reg is None:
            return grad
        elif self.reg == "l1":
            return grad + (self.l1_coef * np.sign(self.weights))
        elif self.reg == "l2":
            return grad + (self.l2_coef * 2 * self.weights)
        elif self.reg == "elasticnet": 
            return grad + (self.l1_coef * np.sign(self.weights)) + (self.l2_coef * 2 * self.weights)

    def get_learning_rate(self, iter):
        if callable(self.learning_rate):
            return self.learning_rate(iter)
        return self.learning_rate        

    def predict_proba(self, X):
        X = X.copy()
        X.insert(0, 'x0', 1)

        return 1 / (1 + np.exp(- ( X @ self.weights)))

    def predict(self, X):
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba > 0.5).astype(int)

    def fit(self, X, y, verbose=False):
        
        random.seed(self.random_state)

        X = X.copy()
        X.insert(0, 'x0', 1)

        n = X.shape[0]
        num_features = X.shape[1]

        self.weights = np.ones(num_features)

        y_pred = 1 / (1 + np.exp(-(X @ self.weights)))
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

            y_pred = 1 / (1 + np.exp(-(X @ self.weights)))

            grad = self.calc_grad(y,y_pred,X, sample_rows_idx)

            current_lerning_rate = self.get_learning_rate(i)

            self.weights = self.weights - current_lerning_rate * grad

            if verbose and i % verbose == 0:
                y_pred = 1 / (1 + np.exp(-(X @ self.weights)))
                loss = self.calc_loss(y, y_pred)
                if self.metric:
                    metric_value = self.calc_metric(y, y_pred)
                    print(f"{i} | loss: {loss} | {self.metric}: {metric_value}")
                else:
                    print(f"{i} | loss: {loss}")

        y_pred = 1 / (1 + np.exp(-(X @ self.weights)))
        self.best_score = self.calc_metric(y, y_pred)            
        
    def get_coef(self):
        return self.weights[1:]    

    def get_best_score(self):
        return self.best_score    