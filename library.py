import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score #, balanced_accuracy_score, precision_score, recall_score

#This class maps values in a column, numeric or categorical.
class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):  
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=True):  
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  #fill in the rest below
  def fit(self, X, y = None):
    print(f"Warning: MappingTransformer.fit does nothing.")
    return X
  
  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'OHETransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'OHETransformer.transform unknown column {self.target_column}'
    value = X.copy()
    value = pd.get_dummies(value,
                               prefix=self.target_column,    #your choice
                               prefix_sep='_',     #your choice
                               columns=[self.target_column],
                               dummy_na=False,    #will try to impute later so leave NaNs in place
                               drop_first=True    #will drop Belfast and infer it
                               )
    return value

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
    self.column_list = column_list
    self.action = action

  #fill in rest below
  def fit(self, X, y = None):
    print(f"Warning: MappingTransformer.fit does nothing.")
    return X
  
  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'DropColumnsTransformer.transform expected Dataframe but got {type(X)} instead.'
    value = X.copy()
    if self.action == 'keep':
      for i in value.columns:
        if i not in self.column_list:
          value.drop([i], axis = 1, inplace = True)
    else:
      for i in value.columns:
        if i in self.column_list:
          value.drop([i], axis = 1, inplace = True)
    return value

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer']
    self.target_column = target_column
    self.fence = fence
    
  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'DropColumnsTransformer.transform expected Dataframe but got {type(X)} instead.'
    value = X.copy()
    q1 = value[self.target_column].quantile(0.25)
    q3 = value[self.target_column].quantile(0.75)
    iqr = q3-q1 
    outer_low = q1-(3*iqr)
    outer_high = q3+(3*iqr)
    inner_low = q1-(1.5*iqr)
    inner_high = q3+(1.5*iqr)
    if self.fence == 'outer':
      value[self.target_column] = value[f'{self.target_column}'].clip(lower=outer_low, upper=outer_high)
    else:
      value[self.target_column] = value[f'{self.target_column}'].clip(lower=inner_low, upper=inner_high)
    return value

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
    
class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):  
    self.target_column = target_column

  #fill in rest below
  def fit(self, X, y = None):
    print(f"Warning: MappingTransformer.fit does nothing.")
    return X
  
  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'DropColumnsTransformer.transform expected Dataframe but got {type(X)} instead.'
    value = X.copy()
    minb, maxb = self.compute_3sigma_boundaries(value, self.target_column)
    if minb < value[self.target_column].min():
      minb = value[self.target_column].min()
    value[self.target_column] = value[f'{self.target_column}'].clip(lower=minb, upper=maxb)
    return value

  def compute_3sigma_boundaries(self, df, column_name):
    #compute mean of column - look for method
    m = df[column_name].mean()
    #compute std of column - look for method
    sigma = df[column_name].std()  
    return  (m-3*sigma, m+3*sigma) #(lower bound, upper bound)

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    self.count = 0

  #fill in rest below
  def fit(self, X, y = None):
    print(f"Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    value = X.copy()
    for i in value[:]:
      mi = value[i].min()
      mx = value[i].max()
      denom = (mx-mi)
      value[i] -= mi
      value[i] /= denom
    return value


  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
 

class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,n_neighbors=5, weights="uniform", add_indicator=False):
    self.n_neighbors = n_neighbors
    self.weights=weights 
    self.add_indicator=add_indicator

  #your code
  def fit(self, X, y = None):
    print(f"Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    value = X.copy()
    imputer = KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False)  #do not add extra column for NaN
    imputed_data = imputer.fit_transform(value)
    result = pd.DataFrame(imputed_data)
    return result


  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  

model = LogisticRegressionCV(max_iter=5000)

def find_random_state(df, labels, n=200):
  var = []
  for i in range(1, n):
    train_X, test_X, train_y, test_y = train_test_split(df, labels, test_size=0.2, shuffle=True,
                                                    random_state=i, stratify=labels)
    model.fit(train_X, train_y)                  #train model
    train_pred = model.predict(train_X)          #predict against training set
    test_pred = model.predict(test_X)            #predict against test set
    train_error = f1_score(train_y, train_pred)  #how bad did we do with prediction on training data?
    test_error = f1_score(test_y, test_pred)     #how bad did we do with prediction on test data?
    error_ratio = test_error/train_error         #take the ratio
    var.append(error_ratio)
  rs_value = sum(var)/len(var)
  abs_val = var - rs_value
  return np.array(abs(abs_val)).argmin() 

titanic_transformer = Pipeline(steps=[
    ('drop', DropColumnsTransformer(['Age', 'Gender', 'Class', 'Joined', 'Married',  'Fare'], 'keep')),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', MappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe', OHETransformer(target_column='Joined')),
    ('age', TukeyTransformer(target_column='Age', fence='outer')), #from chapter 4
    ('fare', TukeyTransformer(target_column='Fare', fence='outer')), #from chapter 4
    ('minmax', MinMaxTransformer()),  #from chapter 5
    ('imputer', KNNTransformer())  #from chapter 6
    ], verbose=True)

customer_transformer = Pipeline(steps=[
    ('id', DropColumnsTransformer(column_list=['ID'])),
    ('os', OHETransformer(target_column='OS')),
    ('isp', OHETransformer(target_column='ISP')),
    ('level', MappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('time spent', TukeyTransformer('Time Spent', 'inner')),
    ('minmax', MinMaxTransformer())
    ], verbose=True)

