import os
import yaml
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_log_error

from src.model import Model
from src.dataset import Dataset

print('Load config file...')
with open('config.yml') as file:
  config = yaml.safe_load(file)

print('Load input data...')
data = {}
for dtype in ['train', 'test']:
  path = os.path.join(config['paths']['inputs'], f'{dtype}.csv')

  data[dtype] = Dataset(
    path=path,
    index_col='id',
    date_col='date',
    factor_cols=['country', 'store', 'product'],
    target_col='num_sold' if dtype == 'train' else None
  )

submission = pd.DataFrame()
for country in data['train'].content['country'].unique():
  print('Make predictions for country : ', country)
  X_train, y_train = data['train'].get_country(country)
  X_test = data['test'].get_country(country)
  
  X_train, X_val, y_train, y_val = train_test_split(
      X_train, y_train, test_size=0.2, random_state=42
  )

  def lgbm_objective(trial):
      lgbm_params = {
          "subsample": trial.suggest_float("subsample", 0.3, 0.9),
          "min_child_samples": trial.suggest_int("min_child_samples", 60, 100),
          "max_depth": trial.suggest_int("max_depth", 7, 25),
          "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
          "lambda_l1": trial.suggest_float("lambda_l1", 0.001, 0.1),
          "lambda_l2": trial.suggest_float("lambda_l2", 0.001, 0.1),
          'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0)
      }
      lgbm_model = Model(lgbm_params)
      lgbm_model.train(X_train, y_train)
      y_pred = lgbm_model.predict(X_val)
      return mean_absolute_percentage_error(y_val, y_pred)

  print('Start optuna study...')
  study_LGBM = optuna.create_study(study_name="LGBM_Kaggle", direction="minimize")
  optuna.logging.set_verbosity(optuna.logging.WARNING)
  study_LGBM.optimize(lgbm_objective, n_trials=200, show_progress_bar=True)

  print("Best trial:", study_LGBM.best_trial)
  print("Best parameters:", study_LGBM.best_params)

  model = Model(study_LGBM.best_params)
  model.train(X_train, y_train)
  y_pred = model.predict(X_val)
  print("MAPE:",mean_absolute_percentage_error(y_val, y_pred))
  y_test = model.predict(X_test)

  sub_country = pd.DataFrame({
      'id': X_test.index,
      'num_sold': y_test
  })
  
  submission = pd.concat([submission, sub_country])
  
submission_path = os.path.join(config['paths']['results'], 'submission.csv')
submission.to_csv(submission_path, index=False)
