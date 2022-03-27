import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    res = np.exp(x)
    res /= res.sum(axis=0)
    
    
    return res


class MulticlassRegression():
    def __init__(self, n_features, n_classes, lr = 0.1):
        self.k = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))
        self.lr = lr
        
    def forward(self, x):
        return softmax(x @ self.k + self.b)
    
    def backward(self, x, y, y_h):
        J_k =  x.T @ (y_h - y)
        self.k -= J_k * self.lr
        
        J_b = (y_h - y).mean(axis=0)
        self.b -= J_b * self.lr
        
    def loss(self, y, y_h):
        L = (-y * np.log(y_h)).mean()
        return L.squeeze()
    
    def train(self, x, y):
        y_h = self.forward(x)
        L = self.loss(y, y_h)
        _ = self.backward(x, y, y_h)
        return L
    
    def evaluate(self, x, y):
        y_h_ohe = self.forward(x)
        y_h = y_h_ohe.argmax(axis=1)
        correct = (y_h==y)
        precision = correct.sum() / len(y_test)
        return precision
    