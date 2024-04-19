import csv
import pandas as pd
import json
from functools import partial
import os
import random

print_now = partial(print, flush=True)

def get_column_names(csv_file, delimiter=','):
  df = pd.read_csv(csv_file, delimiter=delimiter, nrows=2)
  return df.columns

def read_json(path):
  with open(path, 'r') as file:
    data = json.load(file)
  return data  

def shuffle_json(input_file_path, output_file_path):
    def index_file(file_path):
      index = []
      offset = 0
      with open(file_path, 'rb') as file:  # Use binary mode to handle bytes accurately
          while line := file.readline():
              index.append((offset, len(line)))
              offset += len(line)
      return index

    # Index the lines in the original file
    indices = index_file(input_file_path)

    # Shuffle the indices
    random.shuffle(indices)

    # Read lines in the order of shuffled indices and write to new file
    with open(input_file_path, 'rb') as input_file:  # Ensure binary mode for accurate seeking
        with open(output_file_path, 'wb') as output_file:  # Binary mode for output
            for start, length in indices:
                input_file.seek(start)
                line = input_file.read(length)
                output_file.write(line)