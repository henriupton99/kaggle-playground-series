import lightgbm as lgb
import pandas as pd
import numpy as np

class Model:
  def __init__(self, hparams: dict):
    self.hparams = hparams
    
    self.model = lgb.LGBMRegressor(**self.hparams, random_state=42, verbose=-1)
    
  def train(self, X_train: pd.DataFrame, y_train: pd.Series):
    self.model.fit(X_train, y_train)
  
  def predict(self, X_test: pd.DataFrame):
    return np.exp(self.model.predict(X_test))
