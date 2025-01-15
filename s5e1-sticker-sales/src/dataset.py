import os
import pandas as pd
from dataclasses import dataclass

@dataclass
class Dataset:
  path: str
  index_col: str
  date_col: list[str]
  factor_cols: list[str]
  target_col: str|None
  #lag_days: list[int]|None
  
  def __post_init__(self):
    self.dtype = 'test' if self.target_col is None else 'train'
    self.content = pd.read_csv(self.path, parse_dates=[self.date_col], index_col=self.index_col)
    for colname in self.factor_cols:
      self.content[colname] = pd.factorize(self.content[colname])[0]
      #for lag in self.lag_days:
      #  self.content[f'lag_target_{lag}'] = self.content[self.target_col].shift(lag)
    self.content['doy'] = self.content[self.date_col].dt.dayofyear
    self.content['mon'] = self.content[self.date_col].dt.month
    self.content['dow'] = self.content[self.date_col].dt.dayofweek
    self.content['woy'] = self.content[self.date_col].dt.isocalendar().week
    self.content['is_we'] = self.content['dow'].isin([5, 6]).astype(int)
    
    #for window in [7, 30]:
      #train_dataset.features[f'rolling_mean_{window}'] = train_dataset.targets.shift(1).rolling(window=window).mean()
      #train_dataset.features[f'rolling_std_{window}'] = train_dataset.targets.shift(1).rolling(window=window).std()

    self.features = self.content.drop([self.date_col], axis=1)
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
