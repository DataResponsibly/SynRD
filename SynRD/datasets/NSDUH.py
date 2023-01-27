import os
import urllib.request
import zipfile

import pandas as pd

from .dataset import Dataset
from .helpers import URLRetrieveTQDM


class NSDUH(Dataset):
    def __init__(self):
        self.url = "https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/NSDUH-2019/NSDUH-2019-datasets/NSDUH-2019-DS0001/NSDUH-2019-DS0001-bundles-with-study-info/NSDUH-2019-DS0001-bndl-data-tsv.zip"
        self.name = "NSDUH"

    def process(self):
        zip_path = f"data/{self.name}.zip"
        if not os.path.exists(zip_path):
            print("Downloading zip...")
            urllib.request.urlretrieve(self.url, zip_path, URLRetrieveTQDM())

        csv_path = "data/NSDUH_2019_Tab.txt"
        if not os.path.exists(csv_path):
            print("Extracting zip...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall("data")

        print("Reading dataset")
        df = pd.read_csv(csv_path, sep="\t")
        assert isinstance(df, pd.DataFrame)
        return df


class NSDUH_preprocess_Potato(Dataset):
    def __init__(self):
        self.name = "NSDUH_preprocess_Potato"

    def process(self):
        print("Very long preprocessing")
        import time

        # Oopsie
        time.sleep(5)
        df = NSDUH().get()
        df = df.assign(vegetable="potato")
        return df
