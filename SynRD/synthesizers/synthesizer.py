import os
import pandas as pd
import logging

from snsynth.preprocessors import GeneralTransformer, BaseTransformer
from snsynth.pytorch import PytorchDPSynthesizer
from snsynth.pytorch.nn import PATECTGAN


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

class PATECTGAN(Synthesizer):
    def __init__(self, epsilon: float, domain_path: str, slide_range: int = None):
        preprocess_factor = 0.1
        self.synthesizer = PytorchDPSynthesizer(epsilon=epsilon, PATECTGAN(preprocess_epsilon=(preprocess_factor * epsilon)), preprocessor = None)
        super().__init__(epsilon, domain_path, slide_range)
        
    def fit(self, paper: pd.DataFrame):
        if paper.isnull().any().any():
            paper = paper.fillna(0)
            
        df_patectgan = paper[paper.columns].round(0).astype(int)

        self.synthesizer.fit(df_patectgan, categorical_columns=list(df_patectgan.columns), transformer=BaseTransformer)
        sample_size = len(df_patectgan)
        patectgan_synth_data = self.synthesizer.sample(sample_size)

        if self.slide_range:
            paper, range_transform = self.slide_range_forward(paper)
            self.range_transform = range_transform
    
    def sample(self, n, slide_range=None):
        patectgan_synth_data = self.synthesizer.sample(n)
        
        if slide_range:
            patectgan_synth_data = self.slide_range_backward(patectgan_synth_data, self.range_transform)

        return patectgan_synth_data

        # patectgan_synth_data.to_pickle(folder_name + 'patectgan_' + str(it) + '.pickle')
        
class PrivBayes(Synthesizer):
    def __init__(self, epsilon: float, domain_path: str, slide_range: int = None) -> None:
        self.synthesizer = PrivBayes(epsilon=epsilon, domain_path=domain_path)
        super().__init__(epsilon, domain_path, slide_range)
        candidate_keys = {'index': True}
        threshold_value = 40
