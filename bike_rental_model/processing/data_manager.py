import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import typing as t
import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from bike_rental_model import __version__ as _version
from bike_rental_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


##  Pre-Pipeline Preparation

# 1. Preprocess



    
# 2. get year and month

def get_year_and_month(dataframe):

    df = dataframe.copy()
    # convert 'dteday' column to Datetime datatype
    #print(df.head())
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')
    print(df['dteday'])
    # Add new features 'yr' and 'mnth
    df['yr'] = df['dteday'].dt.year
    df['mnth'] = df['dteday'].dt.month_name()
    
    return df
  

def pre_pipeline_preparation(bikeshare) :

  unused_colms = ['dteday', 'casual', 'registered']   # unused columns will be removed at later stage
  target_col = ['cnt']

  numerical_features = []
  categorical_features = []

  for col in bikeshare.columns:
      if col not in target_col + unused_colms:
          if bikeshare[col].dtypes == 'float64':
              numerical_features.append(col)
          else:
              categorical_features.append(col)


  print('Number of numerical variables: {}'.format(len(numerical_features)),":" , numerical_features)

  print('Number of categorical variables: {}'.format(len(categorical_features)),":" , categorical_features)

# First in numerical variables
  bikeshare[numerical_features].isnull().sum()

# Now in categorical variables
  bikeshare[categorical_features].isnull().sum()

    # drop unnecessary variables
  bikeshare.drop(labels=unused_colms, axis=1, inplace=True)

  return bikeshare


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    #df=_load_raw_dataset(file_name="bike-sharing-dataset.csv")
    #transformed = pre_pipeline_preparation(df)

    return get_year_and_month(dataframe)


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    #remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
