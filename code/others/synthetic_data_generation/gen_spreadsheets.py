import pandas as pd
import os
import sys
import numpy as np
import json
from tqdm import tqdm
import logging 


PII_COLS = ['RINPERSOON, RINADRES, BEID, BRIN']

def process_numeric_column(name, data):
  ret_dict = {}
  tenth, q1, q2, q3, ninetieth = np.nanpercentile(data, [10, 25, 50, 75, 90])
  ret_dict[name] = {
    'median': float(np.nanmedian(data)),
    'mean': float(np.nanmean(data)),
    'std_dev': float(np.nanstd(data)),
    '10th_percentile': float(tenth),
    '90th_percentile': float(ninetieth),
    'q1': float(q1),
    'q3': float(q3),
    'null_fraction': float(data.isna().sum() / len(data))
  }
  return ret_dict

def process_categorical_column(name, data):
  ret_dict = {
    name: {}
  }
  for category in data.unique():
    ret_dict[name][category] = float((data==category).sum()/len(data))
  ret_dict[name]['null_fraction'] = float(data.isna().sum()/len(data))
  return ret_dict

def gen_meta_data(df, source_file_path):
  metadata = {
    'path': source_file_path,
    'shape': df.shape,
  }
  columns_with_dtypes = {}
  numeric_data = []
  numeric_columns = []
  for i, column in enumerate(df.columns):
    columns_with_dtypes[column] = str(df.dtypes[i])
    if np.issubdtype(df[column].dtype, np.number):
      numeric_data.append(df[column].tolist())
      numeric_columns.append(column)
  metadata['columns_with_dtypes'] = columns_with_dtypes
  if df.shape[0]<=1:
    return metadata
  if len(numeric_data) > 0:
    numeric_data = np.array(numeric_data).T
    if len(numeric_columns) == 1:
      metadata['cov_matrix'] = np.reshape(np.cov(numeric_data, rowvar=False), (1,1)).tolist()
    else:
      metadata['cov_matrix'] = np.cov(numeric_data, rowvar=False).tolist()
    metadata['numeric_columns'] = numeric_columns
    assert(len(metadata['cov_matrix']) == len(numeric_columns))
  return metadata

def process(source_file_path, target_file_path):
  logging.debug("Starting with file %s.", source_file_path)
  df = pd.read_csv(source_file_path, sep=None, engine="python")
  summary_dict = {
    'metadata': gen_meta_data(df, source_file_path)
  }
  has_pii_cols = []
  logging.debug("Processing columns")
  for column in df.columns:
    logging.debug("current column: %s", column)
    if column in PII_COLS:
      has_pii_cols.append(column)
      continue

    if np.issubdtype(df[column].dtype, np.number):
      summary_dict.update(process_numeric_column(column, df[column]))
    elif df[column].dtype=='object':
      summary_dict.update(process_categorical_column(column, df[column]))
    else:
      logging.warning(
        f"Ignoring {column} from {source_file_path}; "
        f"dtype of {column} is {df[column].dtype}"
      )
  
  summary_dict['has_pii_columns'] = has_pii_cols
  logging.debug("Writing output")
  with open(target_file_path, 'w') as f:
    try:
      json.dump(summary_dict, f)
    except Exception as e:
      logging.error(
        f"{target_file_path} could not be jsonified", 
        exc_info=True,
      )

def create_dir(path):
  pass
  if os.path.exists(path):
    return
  os.mkdir(path)


if __name__ == '__main__':

  logging.basicConfig(
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
  )
  
  root_dir = sys.argv[1]
  target_dir = sys.argv[2]
  for source_root, dirs, files in os.walk(root_dir):
    target_root = source_root.replace(root_dir, target_dir)
    create_dir(target_root)
    
    for f in tqdm(files):
      if f.endswith('.csv'):
        source_path = os.path.join(source_root, f)
        target_path = os.path.join(target_root, f.split('.csv')[0]) + '.txt'
        process(source_path, target_path)
        
