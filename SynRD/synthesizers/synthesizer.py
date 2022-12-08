import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Synthesizer:

    def __init__(self, epsilon: float, domain_path: str, slide_range: int = None) -> None:
        self.data = None
        self.epsilon = epsilon
        self.domain_path = domain_path
        self.slide_range = slide_range
        self.range_transform = None

    def fit(self, paper: pd.DataFrame) -> None:
        pass

    def sample(self, n) -> pd.DataFrame:
        pass

    def load(self, file_path):
        return pd.read_pickle(file_path)

    def save(self, data, base_dir=None):
        file_path = os.path.join(base_dir, self.__name__ + str(self.epsilon) + '.pickle')
        data.to_pickle(file_path)

    @staticmethod
    def slide_range_backward(df, transform) -> pd.DataFrame:
        for c in df.columns:
            if c in transform:
                df[c] = df[c] + transform[c]
        return df

    @staticmethod
    def slide_range_forward(df):
        transform = dict()
        for c in df.columns:
            if min(df[c]) > 0:
                transform[c] = min(df[c])
                df[c] = df[c] - min(df[c])
        return df, transform


class MSTSynthesizer(Synthesizer):
    def __init__(self, epsilon: float, domain_path: str, slide_range: int = None):
        self.synthesizer = MSTSynthesizer(epsilon=epsilon, domain_path=domain_path)
        super().__init__(epsilon, domain_path, slide_range)

    def fit(self, paper: pd.DataFrame):

        if self.slide_range:
            paper, range_transform = self.slide_range_forward(paper)
            self.range_transform = range_transform

        self.synthesizer.fit(paper)

    def sample(self, n, slide_range=None):
        mst_synth_data = self.synthesizer.sample(n)

        if slide_range:
            mst_synth_data = self.slide_range_backward(mst_synth_data, self.range_transform)

        return mst_synth_data
