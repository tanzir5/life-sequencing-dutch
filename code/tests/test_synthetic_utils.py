

from others.synthetic_data_generation.utils import check_column_names

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

