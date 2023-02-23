import json
import os
import math
import numpy as np
import pandas as pd

from pymfe.mfe import MFE

# FEATURE_GROUPS = ["general", "statistical", "info-theory", "concept", "itemset", "complexity"]
FEATURE_GROUPS = ["general", "statistical", "info-theory", "itemset"]

MAPPINGS = {
    # "saw2018cross_dataframe.tsv": {"name": "saw2018cross", "target": ""},
    # "lee2021ability_dataframe.tsv": {"name": "lee2021ability", "target": ""},
    "jeong2021math_dataframe.tsv": {"name": "jeong2021math", "target": "TARGET"},
    # "iverson22football_dataframe.tsv": {"name": "iverson22football", "target": ""},
    # "fruiht2018naturally_dataframe.tsv": {"name": "fruiht2018naturally", "target": ""},
    "29621-0001-Data.tsv": {"name": "fairman2019marijuana", "target": "CLASS"},
    "adult.data": {"name": "adult", "target": "income", "columns": ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]},
    "agaricus-lepiota.data": {"name": "mushrooms", "target": "class", "columns": ["class", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]},
}


def load_data(input_file_path, column_names=None):
    if input_file_path.endswith('.tsv'):
        return pd.read_csv(input_file_path, sep='\t')
    if input_file_path.endswith('.data'):
        return pd.read_csv(input_file_path, sep=',\s', names=column_names, index_col=None)
    if input_file_path.endswith('.csv'):
        return pd.read_csv(input_file_path)
    if input_file_path.endswith('.json'):
        return pd.read_json(input_file_path)
    raise ValueError(f'file {input_file_path} is not supported')


def load_dataframes(input_dir='data/papers'):
    dataframes = dict()
    for input_file_path in MAPPINGS:
        input_file_dict = MAPPINGS[input_file_path]
        dataframe = load_data(os.path.join(input_dir, input_file_path), column_names=input_file_dict.get('columns'))
        dataframes[input_file_dict["name"]] = dataframe
    return dataframes


def get_target(dataframe, target, features=None):
    if features is not None:
        features = features + [target]
        dataframe = dataframe[features]
    X = dataframe.drop(columns=[target], errors='ignore').to_numpy()
    y = dataframe[[target]].to_numpy()
    return X, y


def __replace_nans(vals):
    return [None if math.isnan(v) else v for v in vals]


def get_features(dataframes):
    features = dict()
    name2target = {d['name']: d['target'] for d in MAPPINGS.values()}
    name2features = {d['name']: d.get('features') for d in MAPPINGS.values()}
    for dataframe_name, dataframe in dataframes.items():
        X, y = get_target(dataframe, features=name2features[dataframe_name], target=name2target[dataframe_name])
        dataframe_features = dict()
        for group_name in FEATURE_GROUPS:
            mfe = MFE(groups=group_name)
            mfe.fit(X, y)
            features_names, features_vals = mfe.extract()
            dataframe_features[group_name] = dict(zip(features_names, __replace_nans(features_vals)))
        features[dataframe_name] = dataframe_features
    return features


class NpEncoder(json.JSONEncoder):
    """
    based on: https://stackoverflow.com/a/57915246
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        super(NpEncoder, self).default(obj)


def main():
    dataframes = load_dataframes()
    features = get_features(dataframes)
    with open('features.json', 'w') as output_file:
        json.dump(features, output_file, cls=NpEncoder)


if __name__ == '__main__':
    main()
