import lightgbm as lgb
import pandas as pd

class Model:
  def __init__(self, hparams: dict):
    self.hparams = hparams
    self.hparams['task'] = 'train'
    self.hparams['objective'] = 'regression'
    
    self.model = lgb.LGBMRegressor(**self.hparams)
    
    self.evals_result = {}
    
  def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, eval_metric: str = 'l1'):
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=eval_metric, 
            #evals_result=self.evals_result
        )
  
  def predict(self, X_test: pd.DataFrame):
    return self.model.predict(X_test, num_iteration=self.model.best_iteration_)

  def get_progression(self):
        if not self.evals_result:
            raise ValueError("No training history yet.")
        return pd.DataFrame(self.evals_result['valid_0'])
