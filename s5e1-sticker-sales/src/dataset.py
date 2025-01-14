import os
import pandas as pd
from dataclasses import dataclass

@dataclass
class Dataset:
  path: str
  index_col: str
  date_cols: list[str]
  factor_cols: list[str]
  target_col: str|None
  
  def __post_init__(self):
    self.dtype = 'test' if self.target_col is None else 'train'
    self.content = pd.read_csv(self.path, parse_dates=self.date_cols, index_col=self.index_col)
    for colname in self.factor_cols:
      self.content[colname] = pd.factorize(self.content[colname])[0]
    self.features = self.content.drop(self.date_cols, axis=1)
    if self.dtype == 'train':
      self.features, self.targets = self.features.drop([self.target_col], axis=1), self.content[self.target_col]
  
  def __len__(self):
    return len(self.content)
  
  def __repr__(self):
    _repr = f'Dataset type : {self.dtype}\n' +\
      f'Length : {self.__len__()}\n' +\
      f'Features : {list(self.features.columns)}\n'
    return _repr

if __name__ == '__main__':
  data_path = '/Users/henriup/Desktop/kaggle-playground-series/s5e1-sticker-sales/data'
  train_path = os.path.join(data_path, 'train.csv')
  train_df = Dataset(path=train_path, index_col='id', date_cols=['date'], factor_cols=['country', 'store', 'product'], target_col='num_sold')
  print(train_df)
