import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class Dataset:
  path: str
  index_col: str
  date_col: list[str]
  factor_cols: list[str]
  target_col: str|None
  
  def __post_init__(self):
    self.dtype = 'test' if self.target_col is None else 'train'
    self.content = pd.read_csv(self.path, parse_dates=[self.date_col], index_col=self.index_col)
    for colname in self.factor_cols:
      self.content[colname] = pd.factorize(self.content[colname])[0]
    self.content['year'] = self.content[self.date_col].dt.year
    self.content['month'] = self.content[self.date_col].dt.month
    self.content['day_of_year'] = self.content[self.date_col].dt.dayofyear
    
    self.content["year_sin"] = np.sin(2 * np.pi * self.content["year"]/3)
    self.content["year_cos"] = np.cos(2 * np.pi * self.content["year"]/3)
    self.content["month_sin"] = np.sin(2 * np.pi * self.content["month"] / 12.0)
    self.content["month_cos"] = np.cos(2 * np.pi * self.content["month"] / 12.0)
    self.content['day_sin'] = np.sin(2 * np.pi + self.content['day_of_year']  / 365.0)
    self.content['day_cos'] = np.cos(2 * np.pi + self.content['day_of_year'] / 365.0)
    
    self.content = self.content.drop(['year', 'month', 'day_of_year'], axis=1)
    self.features = self.content.drop([self.date_col], axis=1)
    if self.dtype == 'train':
      self.content[self.target_col] = self.content[self.target_col].bfill()
      self.content[self.target_col] = np.log(self.content[self.target_col])
      self.features, self.targets = self.features.drop([self.target_col], axis=1), self.content[self.target_col]
  
  def get_country(self, country):
    indexes_country = self.features[self.features['country'] == country].index
    features = self.features.loc[indexes_country]
    if self.dtype == 'train':
      targets = self.targets.loc[indexes_country]
      return features, targets
    return features
  
  def __len__(self):
    return len(self.content)
  
  def __repr__(self):
    _repr = f'Dataset type : {self.dtype}\n' +\
      f'Length : {self.__len__()}\n' +\
      f'Features : {list(self.features.columns)}\n'
    return _repr
