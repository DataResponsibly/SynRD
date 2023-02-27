import numpy as np
import pandas as pd
import dill
import os

def _class_to_papername(c):
    return str(c.__name__).lower()

def _int_uniform_sample(lower, upper):
    return int(np.random.uniform(lower, upper))

def bin_df_return_transform(df, columns, num_bins):
    column_transformations = {}
    for column in columns:
        new_column, bins = pd.qcut(df[column], q=num_bins, retbins=True, duplicates='drop')
        column_transformations[column] = bins
        df[column] = new_column.cat.codes
    return df, column_transformations

def unbin_df(df, column_transformations):
    for column, bins in column_transformations.items():
        new_column = df[column]
        df[column] = [_int_uniform_sample(bins[val],bins[val+1]) for val in new_column]
    return df

def do_binning(df, num_bins=10, size_thresh=20):
    cols_to_bin = []
    for col in df.columns:
        max_val = np.max(df[col])
        if max_val > size_thresh:
            cols_to_bin.append(col)
    if len(cols_to_bin) > 0:
        return bin_df_return_transform(df, cols_to_bin, num_bins)
    else:
        return df, None
    
def calculate_domain_size(df):
    domain_size = 1
    for col in df.columns:
        domain_size *= len(df[col].unique())
    return domain_size

def save_synthesizer(model, data_name=None, base_dir=None):
    if data_name is None:
        data_name = ''
    else: 
        data_name = '_' + data_name
    file_path = os.path.join(base_dir, str(model.__class__.__name__) + '_' + str(round(model.epsilon,2)) + data_name + '.pickle')
    dill.dump(model, open(file_path, 'wb'))

def load_synthesizer(file_path):
    return dill.load(open(file_path, 'rb'))