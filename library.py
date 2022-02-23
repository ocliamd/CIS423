import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neighbors import KNeighborsClassifier



# This class maps values in a column, numeric or categorical.
class MappingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, mapping_column, mapping_dict: dict):
        self.mapping_dict = mapping_dict
        self.mapping_column = mapping_column  # column to focus on

    def fit(self, X, y=None):
        print(f"Warning: MappingTransformer.fit does nothing.")
        return X

    def transform(self, X):
        assert isinstance(X,
                          pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
        X_ = X.copy()
        X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
        return X_

    def fit_transform(self, X, y=None):
        result = self.transform(X)
        return result


class RenamingTransformer(BaseEstimator, TransformerMixin):
    # your __init__ method below

    def __init__(self, mapping_dict: dict):
        self.mapping_dict = mapping_dict

    # write the transform method without asserts. Again, maybe copy and paste from MappingTransformer and fix up.
    def transform(self, X):
        assert isinstance(X,
                          pd.core.frame.DataFrame), f'RenamingTransformer.transform expected Dataframe but got {type(X)} instead.'
        # your assert code below

        column_list = X.columns.to_list()
        not_found = list(set(self.mapping_dict.keys()) - set(column_list))
        assert len(
            not_found) < 0, f"Columns {str(not_found)[1:-1]}, are not in the data table"

        X_ = X.copy()
        return X_.rename(columns=self.mapping_dict)


class OHETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, dummy_na=False, drop_first=True):
        self.target_column = target_column
        self.dummy_na = dummy_na
        self.drop_first = drop_first

    def transform(self, X):
        assert isinstance(X,
                          pd.core.frame.DataFrame), f"Expected Pandas DF object, got {type(X)} instead"

        temp_df = X.copy()
        temp_df = pd.get_dummies(temp_df,
                                 columns=[self.target_column],
                                 dummy_na=self.dummy_na,
                                 drop_first=self.drop_first)

        return temp_df

    def fit(self, X):
        return X

    def fit_transform(self, X, X2=None):
        return self.transform(X)


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_list, action='drop'):
        assert action in ['keep',
                          'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list,
                          list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
        self.column_list = column_list
        self.action = action

    def fit(self, X):
        print("Fit does nothing")
        return X

    def transform(self, X):

        temp_df = X.copy()
        if self.action == "drop":
            return temp_df.drop(columns=self.column_list)
        else:
            return temp_df[self.column_list]
            # return temp_df

    def fit_transform(self, X, X2=None):
        return self.transform(X)


class PearsonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, X2=None):
        print("From PearsonTransformer: Warning: Fit does nothing!")
        return X

    def transform(self, X):
        temp = X.copy()
        df_corr = temp.corr(method='pearson')
        masked_df = df_corr.abs() > self.threshold
        upper_mask = np.triu(masked_df, 1)
        t = np.any(upper_mask, 0)
        correlated_columns = [masked_df.columns[i] for i, j in
                              enumerate(upper_mask) if t[i]]
        new_df = transformed_df.drop(correlated_columns, axis=1)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class Sigma3Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column):
        self.target_column = target_column

    def transform(self, X):
        assert isinstance(X,
                          pd.core.frame.DataFrame), f"From Sigma3Transformer: transform expected Pandas DF, got {type(X)}"

        temp = X.copy()
        mean = temp[self.target_column].mean()
        std = temp[self.target_column].std()  # sigma
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        temp[self.target_column] = temp[self.target_column].clip(
            lower=lower_bound, upper=upper_bound)

        return temp

    def fit(self, X, X2=None):
        print("Warning: Sigma3Transformer fit does nothing!")
        return X

    def fit_transform(self, X, X2=None):
        return self.transform(X)


class TukeyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, fence="outer"):
        assert fence in ["inner", "outer"]
        self.target_column = target_column
        self.fence = fence

    def transform(self, X, X2=None):
        temp = X.copy()
        q1 = temp[self.target_column].quantile(0.25)
        q3 = temp[self.target_column].quantile(0.75)
        iqr = q3 - q1
        outer_low = q1 - 3 * iqr
        outer_high = q3 + 3 * iqr
        inner_low = q1 - (1.5 * iqr)
        inner_high = q3 + (1.5 * iqr)

        if self.fence == "inner":
            temp[self.target_column] = temp[self.target_column].\
                clip(lower=inner_low, upper=inner_high)
        else:
            temp[self.target_column] = temp[self.target_column].\
                clip(lower=outer_low, upper=outer_high)
        return temp

    def fit(self, X, X2=None):
        print("Warning: TukeyTransformer fit does nothing!")
        return X

    def fit_transform(self, X, X2=None):
        return self.transform(X)


class MinMaxTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, X2=None):
        print("MinMaxTransformer fit does nothing!")
        return X

    def transform(self, X, X2=None):
        temp = X.copy()
        for col in temp.columns:
            mi = temp[col].min()
            mx = temp[col].max()
            denom = mx - mi
            temp[col] -= mi
            temp[col] /= denom

        return temp

    def fit_transform(self, X, X2=None):
        return self.transform(X)


class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, n_neighbors=5, weights="uniform", add_indicator=False):
    self.n_neighbors = n_neighbors
    self.weights = weights
    self.add_indicator = add_indicator

  def fit(self, x):
    print("KNNTransformer does not do anythin")
    return x

  def transform(self, x, x2=None):
    temp = x.copy()
    knn = KNNImputer(n_neighbors=self.n_neighbors,
                     weights=self.weights, add_indicator=self.add_indicator)
    return pd.DataFrame(data=knn.fit_transform(temp), columns=temp.columns)

  def fit_transform(self, x, x2=None):
    return self.transform(x)


def find_random_state(df, labels, n=200):
  # idx = np.array(abs(var - rs_value)).argmin()
  error_list = []
  for i in range(1, n):
    x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, shuffle=True,
                                                    random_state=i, stratify=labels)
    model.fit(x_train, y_train)
    x_train_pred = model.predict(x_train)  # predict against training set
    x_test_pred = model.predict(x_test)  # predict against test set
    train_error = f1_score(y_train, x_train_pred)  # how bad did we do with prediction on training data?
    test_error = f1_score(y_test, x_test_pred) # how bad did we do with prediction on test data?
    error_list.append(test_error / train_error) # take the ratio
  
  rs_value = sum(error_list)/len(error_list)
  return np.array(abs(error_list - rs_value)).argmin()


titanic_transformer = Pipeline(steps=[
    ('drop', DropColumnsTransformer(
        ['Age', 'Gender', 'Class', 'Joined', 'Married',  'Fare'], 'keep')),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', MappingTransformer('Class', {
     'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe', OHETransformer(target_column='Joined')),
    ('age', TukeyTransformer(target_column='Age', fence='outer')),  # from chapter 4
    ('fare', TukeyTransformer(target_column='Fare', fence='outer')),  # from chapter 4
    ('minmax', MinMaxTransformer()),  # from chapter 5
    ('imputer', KNNTransformer())  # from chapter 6
], verbose=True)


customer_transformer = Pipeline(steps=[
    ('id', DropColumnsTransformer(column_list=['ID'])),
    ('os', OHETransformer(target_column='OS')),
    ('isp', OHETransformer(target_column='ISP')),
    ('level', MappingTransformer('Experience Level',
     {'low': 0, 'medium': 1, 'high': 2})),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('time spent', TukeyTransformer('Time Spent', 'inner')),
    ('minmax', MinMaxTransformer()),
    ('imputer', KNNTransformer())
], verbose=True)


def dataset_setup(feature_table, labels, the_transformer, rs=1234, ts=.2):

  X_train, X_test, y_train, y_test = train_test_split(feature_table, labels, test_size=ts, shuffle=True,
                                                      random_state=rs, stratify=labels)

  X_train_transformed = the_transformer.fit_transform(X_train)
  X_test_transformed = the_transformer.fit_transform(X_test)

  x_trained_numpy = X_train_transformed.to_numpy()
  x_test_numpy = X_test_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  y_test_numpy = np.array(y_test)

  return x_trained_numpy, y_train_numpy, x_test_numpy, y_test_numpy


def titanic_setup(titanic_table, transformer=titanic_transformer, rs=88, ts=.2):
  return dataset_setup(titanic_table.drop(columns='Survived'), titanic_table['Survived'].to_list(), transformer, rs=rs, ts=ts)


def customer_setup(customer_table, transformer=customer_transformer, rs=107, ts=.2):
  return dataset_setup(customer_table.drop(columns='Rating'), customer_table['Rating'].to_list(), transformer, rs=rs, ts=ts)


def threshold_results(thresh_list, actuals, predicted):
  result_df = pd.DataFrame(
      columns=['threshold', 'precision', 'recall', 'f1', 'accuracy'])
  for t in thresh_list:
    yhat = [1 if v >= t else 0 for v in predicted]
    #note: where TP=0, the Precision and Recall both become 0
    precision = precision_score(actuals, yhat, zero_division=0)
    recall = recall_score(actuals, yhat, zero_division=0)
    f1 = f1_score(actuals, yhat)
    accuracy = accuracy_score(actuals, yhat)
    result_df.loc[len(result_df)] = {
        'threshold': t, 'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}
  return result_df


def halving_search(model, grid, x_train, y_train, factor=3, scoring='roc_auc'):
  halving_cv = HalvingGridSearchCV(
      model, grid,  # our model and the parameter combos we want to try
      scoring=scoring,  # could alternatively choose f1, accuracy or others
      n_jobs=-1,
      min_resources="exhaust",
      factor=factor,  # a typical place to start so triple samples and take top 3rd of combos on each iteration
      cv=5, random_state=1234,
      refit=True  # remembers the best combo and gives us back that model already trained and ready for testing
  )

  return halving_cv.fit(x_train, y_train)
