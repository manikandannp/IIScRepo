from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



class WeekdayImputer(BaseEstimator, TransformerMixin):
    """Imputer for weekday column using dteday column."""


    def fit(self, X, y=None):
        X['weekday']=X['weekday'].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        X['dteday'] = pd.to_datetime(X['dteday'])  # Convert 'dteday' column to datetime type
        X['weekday'] = X['dteday'].dt.day_name().str[:3]  # Extract the first three letters of the day name
        X['weekday'] =X['weekday'].fillna(X['weekday'].mode()[0])
        return X.drop('dteday', axis=1)  # Drop the 'dteday' column


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X['weathersit']=X['weathersit'].fillna( X['weathersit'].mode()[0])
        if X['weathersit'].isnull == True:
          X['weathersit']='Clear'
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if X['weathersit'].isnull == True:
          X['weathersit']='Clear'
        X['weathersit']=X['weathersit'].fillna(X['weathersit'].mode()[0])

        return X

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values: 
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variables: list, lower_bound: dict, upper_bound: dict):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        if not isinstance(lower_bound, dict):
            raise ValueError("lower_bound should be a dictionary")
        if not isinstance(upper_bound, dict):
            raise ValueError("upper_bound should be a dictionary")

        self.variables = variables
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # We need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            lower_bound = self.lower_bound.get(feature, None)
            upper_bound = self.upper_bound.get(feature, None)

            if lower_bound is not None:
                X.loc[X[feature] < lower_bound, feature] = lower_bound

            if upper_bound is not None:
                X.loc[X[feature] > upper_bound, feature] = upper_bound

        return X

class WeekdayEncoder(BaseEstimator, TransformerMixin):
    """One-hot encoding for weekday column."""

    def __init__(self, weekday_column: str):
        if not isinstance(weekday_column, str):
            raise ValueError("weekday_column should be a string")

        self.weekday_column = weekday_column

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # We need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        encoded_weekdays = pd.get_dummies(X[self.weekday_column], prefix=self.weekday_column)
        X = pd.concat([X, encoded_weekdays], axis=1).drop(self.weekday_column, axis=1)

        return X
   
class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: list, mappings: dict):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # We need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings[feature])
            X[feature] = X[feature].fillna(np.nan).astype(float)

        return X


    '''def fit(self, X: pd.DataFrame, y: pd.Series=None):
      self.age_avg = X[self.variables].mean()
      self.age_std = X[self.variables].std()
        # we need this step to fit the sklearn pipeline
      return self

    def transform(self, X):
        np.random.seed(42)
    	# so that we do not over-write the original dataframe
        X = X.copy()
        age_null_count = X[self.variables].isnull().sum()
        age_null_random_list = np.random.randint(self.age_avg - self.age_std, self.age_avg + self.age_std, size=age_null_count)
        X.loc[np.isnan(X[self.variables]),self.variables] = age_null_random_list
        X[self.variables] = X[self.variables].astype(int)

        return X'''
    
