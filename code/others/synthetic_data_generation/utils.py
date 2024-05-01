import numpy as np 


def check_column_names(column_names, names_to_check):
    "Check if a colum name matches exactly, or on a substring, the names to check."
    for column_name in column_names:
        for name in names_to_check:
            if name in column_name:
                return True
    return False


def subsample_from_ids(df, id_col="RINPERSOON", frac=0.1):
  """Draw random subsample from dataframe where there are multiple rows
  for `id_col`, but for a given sampled key, all rows should be returned.

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

