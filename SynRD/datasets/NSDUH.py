import os
import urllib.request
import zipfile
from functools import wraps
from typing import Callable, Literal

import pandas as pd
from tqdm.auto import tqdm


def cache_to_pkl(f: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    @wraps(f)
    def wrapper(*args, **kwargs) -> pd.DataFrame:
        args_str = "_".join(map(str, args))
        kwargs_str = "_".join((f"{k}:{v}" for k, v in kwargs.items()))

        name = "_".join(filter(bool, (f.__name__, args_str, kwargs_str)))
        df_path = os.path.join("data", f"{name}_df.pkl")
        if not os.path.isfile(df_path):
            df = f(*args, **kwargs)
            df.to_pickle(df_path)

        df = pd.read_pickle(df_path)
        return df

    return wrapper


@cache_to_pkl
def _load_dataset() -> pd.DataFrame:
    print("Reading dataset")
    data_path = "data/NSDUH_2019_Tab.txt"
    df = pd.read_csv(data_path, sep="\t")
    assert isinstance(df, pd.DataFrame)
    return df


@cache_to_pkl
def _very_important_preprocessing() -> pd.DataFrame:
    print("Very long preprocessing")
    import time

    # Oopie
    time.sleep(5)
    df = _load_dataset()
    df = df.assign(vegetable="potato")
    return df


class URLRetrieveTQDM:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size)

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.n = downloaded
            self.pbar.refresh()
        else:
            self.pbar.close()
            self.pbar = None


def ensure_downloaded():
    zip_path = "data/NSDUH-2019-DS0001-bndl-data-tsv.zip"
    if not os.path.exists(zip_path):
        print("Downloading zip...")
        url = "https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/NSDUH-2019/NSDUH-2019-datasets/NSDUH-2019-DS0001/NSDUH-2019-DS0001-bundles-with-study-info/NSDUH-2019-DS0001-bndl-data-tsv.zip"
        urllib.request.urlretrieve(url, zip_path, URLRetrieveTQDM())

    csv_path = "data/NSDUH_2019_Tab.txt"
    if not os.path.exists(csv_path):
        print("Extracting zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("data")


_func_map = {
    "original": _load_dataset,
    "preprocessed": _very_important_preprocessing,
}


def load(kind: Literal["original", "preprocessed"]):
    ensure_downloaded()
    return _func_map[kind]()


load("preprocessed")
