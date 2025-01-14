import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

from src.model import Model
from src.dataset import Dataset

with open('config.yml') as file:
  config = yaml.safe_load(file)

data = {}
for dtype in ['train', 'test']:
  path = os.path.join(config['paths']['inputs'], f'{dtype}.csv')

  data[dtype] = Dataset(
    path=path,
    index_col='id',
    date_cols=['date'],
    factor_cols=['country', 'store', 'product'],
    target_col='num_sold' if dtype == 'train' else None
  )

X_train, X_val, y_train, y_val = train_test_split(
    data['train'].features, data['train'].targets, test_size=0.2, random_state=42
)
X_test = data['test'].features

model = Model(config['hparams'])
model.train(X_train, y_train, X_val, y_val)

y_pred = model.predict(X_test)

submission = pd.DataFrame({
    'id': data['test'].content.index,
    'num_sold': y_pred
})
submission_path = os.path.join(config['paths']['results'], 'submission.csv')
submission.to_csv(submission_path, index=False)
