import inspect
import os
from abc import ABC, abstractmethod

import pandas as pd

from .helpers import abstract_attribute


class Dataset(ABC):
    name: str = abstract_attribute()

    def __init__(self):
        ...

    @abstractmethod
    def process(self) -> pd.DataFrame:
        ...

    def get(self) -> pd.DataFrame:
        df_path = os.path.join("data", f"{self.name}_df.pkl")
        process_code = inspect.getsource(self.process)
        if not os.path.isfile(df_path):
            df = self.process()
            df.attrs["process"] = process_code
            df.to_pickle(df_path)
        else:
            df = pd.read_pickle(df_path)
            assert df.attrs["process"] == process_code
        return df
