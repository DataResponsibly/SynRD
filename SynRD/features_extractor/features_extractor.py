import json
import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Union, List

from pymfe.mfe import MFE

# FEATURE_GROUPS = ["general", "statistical", "info-theory", "concept", "itemset", "complexity"]
FEATURE_GROUPS = ["general", "statistical", "info-theory", "itemset"]

MAPPINGS = {
    "29621-0001-Data.tsv": {"name": "fairman2019marijuana", "target": "CLASS",
                            "cat_cols": ["YEAR", "CLASS", "SEX", "AGE", "RACE"]},
    "adult.data": {"name": "adult", "target": "income",
                   "columns": ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                               "hours-per-week", "native-country", "income"],
                   "cat_cols": ["workclass", "education", "marital-status", "occupation", "relationship", "race",
                                "sex", "native-country", "income"]},
    "agaricus-lepiota.data": {"name": "mushrooms", "target": "class",
                              "columns": ["class", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor",
                                          "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
                                          "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
                                          "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
                                          "ring-number", "ring-type", "spore-print-color", "population", "habitat"],
                              "cat_cols": "all"},
    "saw2018cross_dataframe.tsv": {"name": "saw2018cross", "target": "stem_career_aspirations", "cat_cols": "all"},
    "lee2021ability_dataframe.tsv": {"name": "lee2021ability", "target": "math", "cat_cols": "auto"},
    "jeong2021math_dataframe.tsv": {"name": "jeong2021math", "target": "TARGET", "cat_cols":  "auto"},
    "iverson22football_dataframe.tsv": {"name": "iverson22football", "target": "H5ID6G"},
    "fruiht2018naturally_dataframe.tsv": {"name": "fruiht2018naturally", "target": "EDU_ATTAINED"},
}

PAPER2TARGET = {d['name']: d['target'] for d in MAPPINGS.values()}
PAPER2FEATURES = {d['name']: d.get('features') for d in MAPPINGS.values()}
PAPER2CAT_COLS = {d['name']: d.get('cat_cols', 'auto') for d in MAPPINGS.values()}


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


def features_target_split(dataframe, target, features=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if features is not None:
        features = features + [target]
        dataframe = dataframe[features]
    X = dataframe.drop(columns=[target], errors='ignore')
    y = dataframe[[target]]
    return X, y


def get_categorical_columns(dataframe_name: str, X: pd.DataFrame) -> Union[str, List[int]]:
    cat_cols = PAPER2CAT_COLS[dataframe_name]
    if cat_cols == 'all':
        return [i for i in range(len(X.columns))]
    if cat_cols == 'auto':
        return cat_cols
    return [i for i, c in enumerate(X.columns) if c in cat_cols]


def __replace_nans(vals):
    return [None if math.isnan(v) else v for v in vals]


class NpEncoder(json.JSONEncoder):
    """
    based on: https://stackoverflow.com/a/57915246
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        super(NpEncoder, self).default(obj)


def get_features_from_files():
    features = dict()
    for peper_name in PAPER2TARGET.keys():
        file_name = f'features_{peper_name}.json'
        features[peper_name] = json.load(open(file_name))
    return features


def calculate_features(dataframe_name, dataframe):
    X, y = features_target_split(dataframe, features=PAPER2FEATURES[dataframe_name],
                                 target=PAPER2TARGET[dataframe_name])
    dataframe_features = dict()
    for group_name in FEATURE_GROUPS:
        mfe = MFE(groups=group_name)
        cat_cols = get_categorical_columns(dataframe_name, X)
        mfe.fit(X.to_numpy(), y.to_numpy(), cat_cols=cat_cols)
        features_names, features_vals = mfe.extract()
        dataframe_features[group_name] = dict(zip(features_names, __replace_nans(features_vals)))
    print(f'Calculated features for {dataframe_name}')
    return dataframe_features


def save_to_file(objs, file_name):
    with open(file_name, 'w') as output_file:
        json.dump(objs, output_file, cls=NpEncoder)
    print(f'Saved to file {file_name}')


def load_from_file(file_name):
    objs = json.load(open(file_name))
    print(f'Loaded from {file_name}')
    return objs


def main():
    dataframes = load_dataframes()
    features = dict()
    for dataframe_name, dataframe in dataframes.items():
        features_file_name = f'features_{dataframe_name}.json'
        if os.path.exists(features_file_name):
            dataframe_features = load_from_file(features_file_name)
        else:
            dataframe_features = calculate_features(dataframe_name, dataframe)
            save_to_file(dataframe_features, features_file_name)
        features[dataframe_name] = dataframe_features
    save_to_file(features, 'features.json')


if __name__ == '__main__':
    main()
