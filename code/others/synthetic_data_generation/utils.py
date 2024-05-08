import numpy as np 
import pyreadstat
import logging
import pandas as pd 

def check_column_names(column_names, names_to_check):
    "Check if a colum name matches exactly, or on a substring, the names to check."
    for column_name in column_names:
        for name in names_to_check:
            if name in column_name:
                return True
    return False


def subsample_from_ids(df, id_col="RINPERSOON", frac=0.1):
  """Draw all rows from a random sample of record ids.

  Args:
    df (pd.DataFrame): dataframe to sample from. 
    id_col (str): column with the identifier. 
    frac (float): Sampling fraction of RINPERSOON records.  
  """
  assert frac > 0 and frac < 1, "frac needs to be between 0 and 1"
  ids = df[id_col].unique()
  n_ids = len(list(ids))
  rng = np.random.default_rng(1234)
  sampled_ids = rng.choice(a=ids, size=int(n_ids*frac), replace=False)
  mask = df[id_col].isin(sampled_ids)
  return df.loc[mask, :]


def sample_from_file(source_file_path, n_rows):
  "Sample n_rows from a file. Return subsampled df and the total number of rows in the file"
  if source_file_path.endswith('.sav'):
    df, _ = pyreadstat.read_sav(source_file_path, metadataonly=True)
    columns = df.columns

    df, _ = pyreadstat.read_sav(source_file_path, usecols=[columns[0]])
    nobs = df.shape[0]

    df, _ = pyreadstat.read_sav(source_file_path, row_limit=n_rows)

  elif source_file_path.endswith(".csv"):
    columns = pd.read_csv(source_file_path, 
                          index_col=0, 
                          nrows=0, 
                          engine="python", 
                          sep=None).columns.tolist()

    df = pd.read_csv(source_file_path, usecols=[columns[0]], engine="python", sep=None)
    nobs = df.shape[0]

    df = pd.read_csv(source_file_path, nrows=n_rows, engine="python", sep=None)
  else:
    raise ValueError(f"wrong file extension found for {source_file_path}")

  return df, nobs
