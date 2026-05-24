class MyKNNClf():
    def __init__(self,
                 k = 3,
                 metric = 'euclidean',
                 weight = 'uniform'):
        self.k = k
        self.train_size = 0
        self.X = None
        self.y = None
        self.metric = metric
        self.weight = weight

    def __repr__(self):
        return f"MyKNNClf class: k={self.k!r}"    

    def fit(self, X, y):
        self.X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)
        self.y = y.to_numpy() if isinstance(y, pd.Series) else np.array(y)

        self.train_size = (self.X.shape[0], self.X.shape[1])

    def calc_metric(self, x_p, x):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((x_p - x)**2))

        if self.metric == "manhattan":
            return np.sum(abs(x_p - x))

        if self.metric == "chebyshev":
            return max(abs(x_p - x))

        if self.metric == "cosine":
            return 1 - np.sum((x_p * x)) / (np.sqrt(np.sum((x_p)**2)) * np.sqrt(np.sum((x)**2)))                   


    def predict(self, X_p):
        X_p = X_p.to_numpy()
        res_pred = []
        for x_p in X_p:
            results = []
            for x in self.X:
                d_metric = self.calc_metric(x_p,x)
                results.append(d_metric)

            min_k_indexes = np.argpartition(results, self.k - 1)[:self.k]
            min_k_indexes = min_k_indexes[np.argsort(np.array(results)[min_k_indexes])]
            min_k_elements = np.array(results)[min_k_indexes]

            if self.weight == 'uniform':

                res = np.mean(self.y[min_k_indexes])

                if res >= 0.5:
                    res_pred.append(1)
                else:
                    res_pred.append(0)  

            if self.weight == 'rank':
                sum_1 = 0
                sum_0 = 0
                sum_all = 0
                y = self.y[min_k_indexes]
                for i in range(self.k):
                    if y[i] == 1:
                        sum_1 += 1/(i+1)
                        sum_all += 1/(i+1)
                    elif y[i] == 0:
                        sum_0 += 1/(i+1)
                        sum_all += 1/(i+1)
                if sum_1/sum_all >= sum_0/sum_all:
                    res_pred.append(1)
                else:
                    res_pred.append(0)         

            if self.weight == 'distance':
                sum_1 = 0
                sum_0 = 0
                sum_all = 0
                y = self.y[min_k_indexes]
                for i in range(self.k):
                    if y[i] == 1:
                        sum_1 += 1/min_k_elements[i]
                        sum_all += 1/min_k_elements[i]
                    elif y[i] == 0:
                        sum_0 += 1/min_k_elements[i]
                        sum_all += 1/min_k_elements[i]
                if sum_1/sum_all >= sum_0/sum_all:
                    res_pred.append(1)
                else:
                    res_pred.append(0)

        return np.array(res_pred)            

        

    def predict_proba(self, X_p):
        X_p = X_p.to_numpy()
        res_pred_prob = []
        for x_p in X_p:
            results = []
            for x in self.X:
                d_metric = self.calc_metric(x_p,x)
                results.append(d_metric)
        
            min_k_indexes = np.argpartition(results, self.k - 1)[:self.k]
            min_k_indexes = min_k_indexes[np.argsort(np.array(results)[min_k_indexes])]
            min_k_elements = np.array(results)[min_k_indexes]

            if self.weight == 'uniform':

                res_pred_prob.append(np.mean(self.y[min_k_indexes]))       

            if self.weight == 'rank':
                sum_1 = 0
                sum_0 = 0
                sum_all = 0
                y = self.y[min_k_indexes]
                for i in range(self.k):
                    if y[i] == 1:
                        sum_1 += 1/(i+1)
                        sum_all += 1/(i+1)
                    elif y[i] == 0:
                        sum_0 += 1/(i+1)
                        sum_all += 1/(i+1)
                res_pred_prob.append(sum_1/sum_all)         

            if self.weight == 'distance':
                sum_1 = 0
                sum_0 = 0
                sum_all = 0
                y = self.y[min_k_indexes]
                for i in range(self.k):
                    if y[i] == 1:
                        sum_1 += 1/min_k_elements[i]
                        sum_all += 1/min_k_elements[i]
                    elif y[i] == 0:
                        sum_0 += 1/min_k_elements[i]
                        sum_all += 1/min_k_elements[i]
                
                res_pred_prob.append(sum_1/sum_all)

        return np.array(res_pred_prob) 