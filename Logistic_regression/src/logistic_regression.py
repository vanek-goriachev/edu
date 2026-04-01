import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression():
    def __init__(self, n_features, lr = 0.1):
        self.k = np.zeros((1, n_features))
        self.b = np.zeros((1, 1))
        self.lr = lr
        
    def forward(self, x):
        return sigmoid(self.k @ x.T + self.b)
    
    def backward(self, x, y, y_h):
        J_k =  2 * (y - y_h) @ -x
        self.k -= J_k * self.lr
        
        J_b =  2 * (y - y_h) * -1 @ np.ones(shape=x.shape[0])
        self.b -= J_b * self.lr
        
    def loss(self, y, y_h):
        
        L = (-(y * np.log(y_h) + (1-y) * np.log(1 - y_h))).mean()
        return L.squeeze()
    
    def train(self, x, y):
        y_h = self.forward(x)
        L = self.loss(y, y_h)
        _ = self.backward(x, y, y_h)
        return L
    
    def evaluate(self, x, y):
        y_h = self.forward(x)
        L = self.loss(y, y_h)
        return L
