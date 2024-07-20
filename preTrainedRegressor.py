from sklearn.base import BaseEstimator, RegressorMixin

class PreTrainedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y):
        # Do nothing, as the model is already trained
        return self
    
    def predict(self, X):
        return self.model.predict(X)
