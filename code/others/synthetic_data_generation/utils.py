

def check_column_names(column_names, names_to_check):
    "Check if a colum name matches exactly, or on a substring, the names to check."
    for column_name in column_names:
        for name in names_to_check:
            if name in column_name:
                return True
    return False

