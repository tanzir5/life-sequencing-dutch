# Set of unit tests for ensuring that the report code continues to function as expected
import numpy as np
import pickle
import report_utils


########################################################################################################################
# Test imports
def import_test():
    from scipy.spatial import distance as dst
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import cross_val_score
    from torch import Tensor
    from sentence_transformers import util
    import matplotlib.pyplot as plt
    import pickle
    import pandas as pd
    import numpy as np
    import random
    import csv
    import json
    import h5py
    import logging

########################################################################################################################
# Test embedding properties
def test_embeddings(embedding_dict, embedding_name):
    # Ensure that embeddings are indexed by integers
    keys = list(embedding_dict.keys())
    for key in keys:
        embedding = embedding_dict[key]
        assert type(key) == int, "Test Failed: embedding dict <" + embedding_name + "> is not indexed by integers - found type " + str(type(key))
        assert type(embedding) == np.ndarray, "Test Failed: embedding dict <" + embedding_name + "> contains a value that is not a Numpy ndarray! - found type " + str(type(embedding))
        assert embedding is not None, "Test Failed: embedding dict <" + embedding_name + "> contains at least one None value!"
        assert not np.isnan(np.sum(embedding)), "Test Failed: embedding dict <" + embedding_name + "> contains at least one NaN value!"

    # Ensure that embeddings are not of length 0 (there must be content)
    first_key = keys[0]
    length = len(embedding_dict[first_key])
    assert length > 0, "Test Failed: embedding dict <" + embedding_name + "> contains an embedding of length 0!"

    # Ensure that all embeddings are the same shape
    shape = embedding_dict[first_key].shape
    for key in keys:
        other_shape = embedding_dict[key].shape
        assert shape == other_shape, "Test Failed: embedding dict <" + embedding_name + "> contains ragged embeddings! " + str(shape) + " : " + str(other_shape)

########################################################################################################################
# These variables are scalars only! Things like income, or death year
def test_single_variable(variable_dict, variable_name):
    # Ensure the variables are indexed by integers
    keys = list(variable_dict.keys())
    for key in keys:
        variable = variable_dict[key]
        assert type(key) == int, "Test Failed: variable dict <" + variable_name + "> is not indexed by integers - found type " + str(type(key))
        assert variable is not None, "Test Failed: variable dict <" + variable_name + "> contains at least one None value"
        assert not np.isnan(variable), "Test Failed: variable dict <" + variable_name + "> contains at least one NaN value"
        assert type(variable) != tuple and type(variable) != list and type(variable) != np.ndarray, "Test Failed: variable dict <" + variable_name + "> contains a variable that is not a scalar - found type " + str(type(variable))

    # Ensure that variables are not of length -
    first_key = keys[0]
    length = len(variable_dict[first_key])
    assert length > 0, "Test Failed: variable dict <" + variable_name + "> contains a variable of length 0!"

    # Ensure that all variables are the same shape
    for key in keys:
        other_length = len(variable_dict[key])
        assert length == other_length, "Test Failed: variable lengths are different! " + str(length) + " : " + str(other_length)

########################################################################################################################
def test_years(years):
    # Ensure that years are represented as integers and that we have at least one year to test
    assert len(years) > 0, "Test Failed: Length of <years> is 0"
    for year in years:
        assert type(year) == int, "Test Failed: at least one instance in <years> is not an integer - found type " + str(type(year))
    # Ensure that we don't have any years past 2022
    max_year = max(years)
    assert max_year <= 2022, "Test Failed: highest value of <years> is " + str(max_year)

########################################################################################################################
def test_overlap(embedding_dict, variable_dict, baseline):

    embedding_persons = set(embedding_dict.keys())
    variable_persons = set(variable_dict.keys())
    baseline_persons = set(baseline.keys())

    all_people = embedding_persons.intersection(variable_persons).intersection(baseline_persons)
    assert len(all_people) > 100, "Test Failed: there are less than 100 people in this intersection"

########################################################################################################################
def test_pair_variable(variable_dict, variable_name):

    # Ensure the variables are indexed by integers
    for pair in variable_dict:
        assert len(pair) == 2, "Test Failed: variable dict <" + variable_name + "> contains tuples that are not pairs"

        person = pair[0]
        partner = pair[1]
        assert type(person) == int and type(partner) == int, "Test Failed: variable dict <" + variable_name + "> includes IDs that are not ints"

########################################################################################################################
def test_baseline(baseline, baseline_name):
    # Ensure the variables are indexed by integers
    keys = list(baseline.keys())
    for key in keys:
        values = baseline[key]
        assert type(key) == int, "Test Failed: baseline <" + baseline_name + "> is not indexed by integers - found type " + str(type(key))

        for value in values:
            assert value is not None, "Test Failed: baseline <" + baseline_name + "> contains at least one None value"
            assert not np.isnan(value), "Test Failed: variable dict <" + baseline_name + "> contains at least one NaN value"

    # Ensure that values are not of length 0
    first_key = keys[0]
    length = len(baseline[first_key])
    assert length > 0, "Test Failed: baseline <" + baseline_name + "> contains a value of length 0!"

    # Ensure that all values are the same shape
    for key in keys:
        other_length = len(baseline[key])
        assert length == other_length, "Test Failed: baseline value lengths are different! " + str(length) + " : " + str(other_length)

########################################################################################################################
# Run tests
if __name__ == '__main__':

    # Load income
    print("Testing income variable...", flush=True)
    income_by_year = report_utils.precompute_global('income')
    years = list(income_by_year.keys())
    test_years(years)

    # Test that each year of the income variable is well formed
    for year in years:
        yearly_income = income_by_year[year]
        test_single_variable(yearly_income, "Income-" + str(year))

    # Load marriages
    print("Testing marriage variable...", flush=True)
    marriages_by_year, partnerships_by_year = report_utils.precompute_global('marriage')
    years = list(marriages_by_year.keys())
    test_years(years)

    # Test that each year of the marriage pair variable is well formed
    for year in years:
        yearly_marriages = marriages_by_year[year]
        test_pair_variable(yearly_marriages, "Marriage-" + str(year))

    ####################################################################################################################
    # Load Naive baseline
    # 1. Birth Year (Age)
    with open("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/processed/person_birth_year.pkl",
              'rb') as pkl_file:
        person_birth_year = dict(pickle.load(pkl_file))

    # 2. Gender
    with open("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/processed/person_gender.pkl", 'rb') as pkl_file:
        person_gender = dict(pickle.load(pkl_file))

    # 3. Birth City
    with open("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/processed/person_birth_municipality.pkl",
              'rb') as pkl_file:
        person_birth_city = dict(pickle.load(pkl_file))

    # Combine the baseline parts into a single dict to pass to evaluation functions
    baseline_dict = {}

    for person in person_birth_year:
        birth_year = person_birth_year[person]
        gender = person_gender[person]
        birth_city = person_birth_city[person]

        baseline_dict[person] = [birth_year, gender, birth_city]

    print("Testing naive baseline...", flush=True)
    test_baseline(baseline_dict, "Naive baseline")

    ####################################################################################################################

    # Try out the tests with the Groningen embeddings
    print("Testing Groningen embeddings...", flush=True)
    load_url = 'embedding_meta/gron_embedding_set.pkl'
    with open(load_url, 'rb') as pkl_file:
        embedding_sets = list(pickle.load(pkl_file))

        for i, emb in enumerate(embedding_sets):

            embedding_dict = report_utils.precompute_local(emb, only_embedding=True)
            years = list(embedding_dict.keys())

            # Make sure the years are normal
            test_years(years)

            for year in years:
                yearly_dict = embedding_dict[year]
                test_embeddings(yearly_dict, "Gron_" + str(i))