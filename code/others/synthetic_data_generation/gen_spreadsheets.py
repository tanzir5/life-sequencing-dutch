import pandas as pd
import os
import sys
import numpy as np
import json
from tqdm import tqdm

def warning_message(df, column, path):
  print(f"Ignoring {column} from {path}")
  print(f"dtype of {column} is {df[column].dtype}")


def process_numeric_column(name, data):
  ret_dict = {}
  tenth, q1, q2, q3, ninetieth = np.nanpercentile(data, [10, 25, 50, 75, 90])
  ret_dict[name] = {
    'min': float(np.nanmin(data)),
    'median': float(np.nanmedian(data)),
    'max': float(np.nanmax(data)),
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
    # print(len(numeric_columns))
    # print(metadata['cov_matrix'])
    # print(numeric_columns)
    # print(numeric_data.shape)
    assert(len(metadata['cov_matrix']) == len(numeric_columns))
  return metadata

def process(source_file_path, target_file_path):
  df = pd.read_csv(source_file_path)
  summary_dict = {
    'metadata': gen_meta_data(df, source_file_path)
  }
  for column in df.columns:
    if np.issubdtype(df[column].dtype, np.number):
      summary_dict.update(process_numeric_column(column, df[column]))
    elif df[column].dtype=='object':
      summary_dict.update(process_categorical_column(column, df[column]))
    else:
      warning_message(df, column, source_file_path)

  with open(target_file_path, 'w') as f:
    try:
      json.dump(summary_dict, f)
    except Exception as e:
      print("An error occurred:", e)
      print(f"{target_file_path} could not be jsonified")
  
def create_dir(path):
  pass
  if os.path.exists(path):
    return
  os.mkdir(path)


if __name__ == '__main__':
  root_dir = sys.argv[1]
  target_dir = sys.argv[2]
  for source_root, dirs, files in os.walk(root_dir):
    target_root = source_root.replace(root_dir, target_dir)
    create_dir(target_root)

    # print("ROOT"*10)
    # print(root)
    # input()
    # print("DIRS"*10)
    # print(dirs)
    # input()
    # print("FILES"*10)
    # print(files)
    # input()
    
    for f in tqdm(files):
      if f.endswith('.csv'):
        source_path = os.path.join(source_root, f)
        target_path = os.path.join(target_root, f.split('.csv')[0]) + '.json'
        process(source_path, target_path)
        
