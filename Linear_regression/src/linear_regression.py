import numpy as np

class LinearRegression():
    def __init__(self, n_features, lr = 0.1):
        self.k = np.zeros((1, n_features))
        self.b = np.zeros((1, 1))
        self.lr = lr
        
    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return self.k @ x.T + self.b
    
    def backward(self, x, y, y_h):
        J_k =  2 * (y - y_h) @ -x
        self.k -= J_k * self.lr
        
        J_b =  2 * (y - y_h) * -1 @ np.ones(shape=x.shape[0])
        self.b -= J_b * self.lr
        
    def loss(self, y, y_h):
        n = y.shape[0]
        L = (y - y_h) @ (y - y_h).T / n
        return L.squeeze()
    
    def train(self, x, y):
        y_h = self.forward(x)
        L = self.loss(y, y_h)
        _ = self.backward(x, y, y_h)
        return L
    
    def evaluate(self, x, y):
        y_h = self.forward(x).squeeze()
        L = self.loss(y, y_h)
        return L
