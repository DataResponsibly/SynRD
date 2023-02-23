import os
import pandas as pd
import logging

from snsynth.pytorch import PytorchDPSynthesizer
from snsynth.pytorch.nn import PATECTGAN as SmartnoisePATECTGAN
from snsynth.mst import MSTSynthesizer as SmartnoiseMSTSynthesizer
from snsynth.aggregate_seeded import AggregateSeededSynthesizer
from snsynth.transform import NoTransformer
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

logger = logging.getLogger(__name__)


class Synthesizer:

    def __init__(self, 
                 epsilon: float, 
                 slide_range: bool = True,
                 thresh = 0.05) -> None:
        self.data = None
        self.epsilon = epsilon
        self.slide_range = slide_range
        self.range_transform = None
        self.thresh = thresh

    def fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def sample(self, n) -> pd.DataFrame:
        raise NotImplementedError

    def load(self, file_path):
        return pd.read_pickle(file_path)

    def save(self, data, base_dir=None):
        file_path = os.path.join(base_dir, self.__name__ + str(self.epsilon) + '.pickle')
        data.to_pickle(file_path)

    def _slide_range(self, df):
        if self.slide_range:
            df, self.range_transform = self.slide_range_forward(df)
        return df

    def _unslide_range(self, df):
        if self.slide_range and self.range_transform is None:
            raise ValueError('Must fit synthesizer before sampling.')
        if self.slide_range:
            df = self.slide_range_backward(df, self.range_transform)
        return df
    
    def _categorical_continuous(self, df):
        # NOTE: return categorical/ordinal columns and continuous
        # This is slightly hacky, but should be fine.
        categorical = []
        continuous = []
        for col in df.columns:
            if (float(df[col].nunique()) / df[col].count()) < self.thresh:
                categorical.append(col)
            else:
                continuous.append(col)
        return {'categorical': categorical, 'continuous':  continuous}

    @staticmethod
    def slide_range_forward(df):
        transform = dict()
        for c in df.columns:
            if min(df[c]) > 0:
                transform[c] = min(df[c])
                df[c] = df[c] - min(df[c])
        return df, transform

    @staticmethod
    def slide_range_backward(df, transform) -> pd.DataFrame:
        for c in df.columns:
            if c in transform:
                df[c] = df[c] + transform[c]
        return df


class MSTSynthesizer(Synthesizer):
    def __init__(self, 
                 epsilon: float, 
                 slide_range: bool = False,
                 thresh = 0.05,
                 preprocess_factor: float = 0.05):
        self.synthesizer = SmartnoiseMSTSynthesizer(epsilon=epsilon)
        self.preprocess_factor = preprocess_factor
        super().__init__(epsilon, slide_range, thresh)

    def fit(self, df: pd.DataFrame):
        categorical_check = (len(self._categorical_continuous(df)['categorical']) == len(list(df.columns)))
        if not categorical_check:
            raise ValueError('Please make sure that MST gets categorical/ordinal\
                features only. If you are sure you only passed categorical, \
                increase the `thresh` parameter.')

        df = self._slide_range(df)
        self.synthesizer.fit(df, preprocessor_eps=(self.preprocess_factor * self.epsilon))

    def sample(self, n):
        df = self.synthesizer.sample(n)
        df = self._unslide_range(df)
        return df

class PATECTGAN(Synthesizer):
    def __init__(self, 
                 epsilon: float, 
                 slide_range: bool = False,
                 preprocess_factor: float = 0.05,
                 thresh = 0.05):
        self.preprocess_factor = preprocess_factor
        self.synthesizer = PytorchDPSynthesizer(epsilon=epsilon, 
                                                gan=SmartnoisePATECTGAN(epsilon=epsilon))
        super().__init__(epsilon, slide_range, thresh)
        
    def fit(self, df: pd.DataFrame):
        df = self._slide_range(df)
        cat_con = self._categorical_continuous(df) 
        self.synthesizer.fit(df, 
                             categorical_columns=cat_con['categorical'], 
                             continuous_columns=cat_con['continuous'],
                             preprocessor_eps=(self.preprocess_factor * self.epsilon))
    
    def sample(self, n):
        df = self.synthesizer.sample(n)
        df = self._unslide_range(df)
        return df
        
class PrivBayes(Synthesizer):
    def __init__(self, 
                 epsilon: float, 
                 slide_range: bool = None,
                 thresh = 0.05,
                 privbayes_limit = 20,
                 privbayes_bins = 10,
                 temp_files_dir = 'temp',
                 seed = 0) -> None:
        self.privbayes_limit = privbayes_limit
        self.privbayes_bins = privbayes_bins
        self.temp_files_dir = temp_files_dir
        self.seed = seed

        self.describer = DataDescriber(category_threshold=self.privbayes_limit)
        self.generator = DataGenerator()

        os.makedirs(self.temp_files_dir, exist_ok=True)
        self.candidate_keys = {'index': True}
        self.dataset_size = None
        super().__init__(epsilon, slide_range, thresh)
        
    def fit(self, df: pd.DataFrame):
        df = self._slide_range(df)

        # NOTE: PrivBayes implementation has some weird requirements
        # as it runs so slowly when data is high dimensional
        # Here, we check to see whether we need to bin data 
        binned = {}
        for col in df.columns:
            if len(df[col].unique()) > self.privbayes_limit:
                col_df = pd.qcut(df[col], q=self.privbayes_bins, duplicates='drop')
                df[col] = col_df.apply(lambda row : row.mid).astype(int)
                binned[col] = True
        
        cat_con = self._categorical_continuous(df) 
        categorical_check = (len(cat_con['categorical']) == len(list(df.columns)))
        if not categorical_check:
            raise ValueError('PrivBayes does not work with continous columns. Suggest \
                decreasing the `privbayes_limit` or increasing the `thresh` parameter.')

        df.to_csv(os.path.join(self.temp_files_dir, "temp.csv"))
        self.dataset_size = len(df)
        self.describer.describe_dataset_in_correlated_attribute_mode(f"{self.temp_files_dir}/temp.csv",
                                                        epsilon=self.epsilon, 
                                                        k=2,
                                                        attribute_to_is_categorical=binned,
                                                        attribute_to_is_candidate_key=self.candidate_keys,
                                                        seed=self.seed)
        self.describer.save_dataset_description_to_file(f"{self.temp_files_dir}/privbayes_description.csv")

    def sample(self, n):
        self.generator.generate_dataset_in_correlated_attribute_mode(n,
                                                                f"{self.temp_files_dir}/privbayes_description.csv")
        self.generator.save_synthetic_data(f"{self.temp_files_dir}/privbayes_synth.csv")
        df = pd.read_csv(f"{self.temp_files_dir}/privbayes_synth.csv", index_col=0)

        df = self._unslide_range(df)
        return df

class PacSynth(Synthesizer):
    def __init__(self, 
                 epsilon: float, 
                 slide_range: bool = None,
                 thresh = 0.05) -> None:
        self.synthesizer = AggregateSeededSynthesizer(epsilon=epsilon,
                                                      percentile_percentage=99,
                                                      reporting_length=3)
        super().__init__(epsilon, slide_range, thresh)

    def fit(self, df: pd.DataFrame):
        df = self._slide_range(df)
        self.synthesizer.fit(df, transformer=NoTransformer())
    
    def sample(self, n):
        df = self.synthesizer.sample(n)
        df = self._unslide_range(df)
        return df
    
class AIMSynthesizer(Synthesizer):
    pass