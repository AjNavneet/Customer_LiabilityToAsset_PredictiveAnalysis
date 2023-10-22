import pandas as pd

# Read data from a CSV file with optional keyword arguments
def read_data(file_path, **kwargs):
    raw_data = pd.read_csv(file_path, **kwargs)
    return raw_data

# Merge two DataFrames based on specified join type and columns
def merge_dataset(df1, df2, join_type, on_param):
    final_df = df1.copy()
    final_df = final_df.merge(df2, how=join_type, on=on_param)
    return final_df

# Drop specified columns from a DataFrame
def drop_col(df, col_list):
    for col in col_list:
        if col not in df.columns:
            raise ValueError(f"Column does not exist in the DataFrame")
        else:
            df = df.drop(col, axis=1)
    return df

# Remove rows with null values from a DataFrame
def null_values(df):
    df = df.dropna()
    return df

# Find the maximum value in a list and return its value and index
def max_val_index(l):
    max_l = max(l)
    max_index = l.index(max_l)
    return (max_l, max_index)
