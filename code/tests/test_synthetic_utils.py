
import pandas as pd 
import numpy as np 

from others.synthetic_data_generation.utils import check_column_names
from others.synthetic_data_generation.utils import subsample_from_ids


def test_check_column_names():
    # Test case 1: Basic functionality
    column_names = ['apple', 'banana', 'grape', 'pineapple', 'watermelon']
    names_to_check = ['apple', 'banana']
    assert check_column_names(column_names, names_to_check)

    # Test case 2: No matches
    column_names = ['kiwi', 'orange', 'pear']
    names_to_check = ['apple', 'banana']
    assert not check_column_names(column_names, names_to_check)

    # Test case 3: Partial matches
    column_names = ['apple', 'banana', 'grape', 'pineapple', 'watermelon']
    names_to_check = ['ppl', 'anana']
    assert check_column_names(column_names, names_to_check) 

    # Test case 4: Empty input lists
    column_names = []
    names_to_check = []
    assert not check_column_names(column_names, names_to_check)


def test_subsample_from_ids():
    df = pd.DataFrame({
        "person": [1,2,3,4,1,2,3,4], 
        "spell": [1,1,1,1,2,2,2,2], 
        "wage": np.random.rand(8)}
    )
    sampled_df = subsample_from_ids(df, id_col="person", frac=0.5)
    assert sampled_df.shape == (4, df.shape[1]), "does not return right shape"
    assert (sampled_df.columns == df.columns).all(), "does not retain column names"
    assert sampled_df["person"].nunique() == 2, "does not sample right fraction of ids"