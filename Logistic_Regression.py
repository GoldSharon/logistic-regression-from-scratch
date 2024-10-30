import numpy as np 

class Logistic_Regression:
    
    def __init__(self,learning_rate=0.01,num_of_iter=1000):
        
        self.learning_rate = learning_rate
        
        self.num_of_iter = num_of_iter
        
        self.weights = None
        
        self.bias = 0
        
        self.rows = None
        
        self.cols = None
        
    def fit(self,X,y):
        
        self.rows , self.cols = X.shape
        
        self.weights = np.zeros((self.cols,1))
        
        for i in range(self.num_of_iter):
            
            self.update_weights(X,y.reshape(-1,1))
            
    
    
    def predict(self,X):
        
        z =  (X @ (self.weights) + self.bias ) 
        
        y_pred = 1/(1+np.exp(-z))
        
        y_pred = np.where(y_pred>0.5,1,0)
        
        return y_pred
        
        
    def update_weights(self, X, y):
        z = (X @ self.weights + self.bias)
        Ý = 1 / (1 + np.exp(-z))
        
        # Compute gradients
        dw = (1 / self.rows) * (X.T @ (Ý - y))  # Gradient w.r.t weights
        db = (1 / self.rows) * np.sum(Ý - y)     # Gradient w.r.t bias
        
        # Update weights and bias
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        
        