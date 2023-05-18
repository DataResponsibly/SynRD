import os
import json
import pandas as pd
import numpy as np

from snsynth.mst import MSTSynthesizer
from snsynth.preprocessors import BaseTransformer
from snsynth.pytorch import PytorchDPSynthesizer
from snsynth.pytorch.nn import PATECTGAN

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator


class PrivateDataGenerator():
    """
    Central data privatizer - generates private data
    in proper folder architecture given a Publication.
    """
    EPSILONS = [(np.e ** -1, 'e^-1'), 
            (np.e ** 0, 'e^0'), 
            (np.e ** 1, 'e^1'),
            (np.e ** 2, 'e^2')]

    ITERATIONS = 5

    # Add a json domain file for each paper here, MST requirement.
    DOMAINS = {
        "saw2018cross": "domains/saw2018cross-domain.json",
        "jeong2021math": "domains/jeong2021math-domain.json",
        "fairman2019marijuana": "domains/fairman2019marijuana-domain.json",
        "fruiht2018naturally": "domains/fruiht2018naturally-domain.json",
        "lee2021ability": "domains/lee2021ability-domain.json",
        "iverson22football": "domains/iverson22football-domain.json",
    }

    def __init__(self, publication, slide_range=False, privbayes_limit=40, privbayes_bins=10):
        self.publication = publication
        self.cont_features = publication.cont_features
        self.slide_range = slide_range
        self.privbayes_limit = privbayes_limit
        self.privbayes_bins = privbayes_bins

    def prepare_dataframe(self):
        df = self.publication.dataframe
        print(df.apply(lambda x: x.unique()))
        print(df.apply(lambda x: len(x.unique())))
        return df

    def slide_range_forward(self, df):
        transform = {}
        for c in df.columns:
            if min(df[c]) > 0:
                transform[c] = min(df[c])
                df[c] = df[c] - min(df[c])
        return df, transform

    def slide_range_backward(self, df, transform):
        for c in df.columns:
            if c in transform:
                df[c] = df[c] + transform[c]
        return df
    
    def generate(self):
        df = self.prepare_dataframe()

        df_map = {
            self.publication.DEFAULT_PAPER_ATTRIBUTES['id'] : df
        }

        if self.slide_range:
            df, range_transform = self.slide_range_forward(df)

        # Threshold binning for larger values for privbayes:
        # NOTE: this is due to time efficiency issues
        df_privbayes = df.copy()
        binned = {}
        for col in df_privbayes.columns:
            if len(df_privbayes[col].unique()) > self.privbayes_limit:
                col_df = pd.qcut(df_privbayes[col], q=self.privbayes_bins, duplicates='drop')
                df_privbayes[col] = col_df.apply(lambda row : row.mid).astype(int)
                binned[col] = True
        print('UNIQUE')
        print(df_privbayes.apply(lambda x: len(x.unique())))
        temp_files_dir = 'temp'
        os.makedirs(temp_files_dir, exist_ok=True)
        df_privbayes.to_csv(os.path.join(temp_files_dir, "temp.csv"))

        for pub_name, df in df_map.items():
            print('Generating: ' + pub_name)
            if not os.path.exists('private_data/' + str(pub_name)):
                os.mkdir('private_data/' + str(pub_name))
            for (eps, str_eps) in self.EPSILONS:
                print(f'EPSILON: {str_eps}...')

                if not os.path.exists('private_data/' + str(pub_name) + '/' + str_eps):
                    os.mkdir('private_data/' + str(pub_name) + '/' + str_eps)

                for it in range(self.ITERATIONS):
                    print(f'ITERATION: {it}...')

                    # Folder for deposit
                    folder_name = 'private_data/' + str(pub_name) + '/' + str_eps + '/'
                    
                    if not os.path.isfile(folder_name + 'mst_' + str(it) + '.pickle'):
                        # The MST Synthesis
                        mst = MSTSynthesizer(epsilon=eps,
                                             domain=pub_name, 
                                             domains_dict=self.DOMAINS)
                        
                        mst.fit(df)
                        sample_size = len(df)
                        mst_synth_data = mst.sample(sample_size)

                        if self.slide_range:
                            mst_synth_data = self.slide_range_backward(mst_synth_data, range_transform)

                        mst_synth_data.to_pickle(folder_name + 'mst_' + str(it) + '.pickle')
                        print(mst_synth_data.apply(lambda x: x.unique()))
                        print(mst_synth_data.apply(lambda x: len(x.unique())))

                    print('DONE: MST.')

                    if not os.path.isfile(folder_name + 'patectgan_' + str(it) + '.pickle'):
                        # The PATECTGAN Synthesis
                        preprocess_factor = 0.1
                        patectgan = PytorchDPSynthesizer(eps, 
                                                        PATECTGAN(preprocessor_eps=(preprocess_factor * eps)), 
                                                        preprocessor=None)
                        # Sadly, patectgan needs this sort of rounding right now
                        if df.isnull().any().any():
                            df = df.fillna(0) 

                        df_patectgan = df[df.columns].round(0).astype(int)
                        patectgan.fit(
                            df_patectgan,
                            categorical_columns=list(df_patectgan.columns),
                            transformer=BaseTransformer,
                        )
                        sample_size = len(df_patectgan)
                        patectgan_synth_data = patectgan.sample(sample_size)

                        if self.slide_range:
                            patectgan_synth_data = self.slide_range_backward(patectgan_synth_data, range_transform)

                        patectgan_synth_data.to_pickle(folder_name + 'patectgan_' + str(it) + '.pickle')
                        print(patectgan_synth_data.apply(lambda x: x.unique()))
                        print(patectgan_synth_data.apply(lambda x: len(x.unique())))

                    print('DONE: PATECTGAN.')

                    if not os.path.isfile(folder_name + 'privbayes_' + str(it) + '.pickle'):
                        # The PrivBayes Synthesis

                        # specify which attributes are candidate keys of input dataset.
                        candidate_keys = {'index': True}

                        # An attribute is categorical if its domain size is less than this threshold.
                        # Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
                        threshold_value = 40

                        domain_name = self.DOMAINS[pub_name]
                        with open(domain_name) as json_file:
                            dict_domain = json.load(json_file)

                        # temp for PrivBayes to show there are cont values
                        if self.cont_features:
                            for cont_feature in self.cont_features:
                                dict_domain[cont_feature] = threshold_value + 1

                        # specify categorical attributes
                        categorical_attributes = {k: True for k, v in dict_domain.items() if v < threshold_value}
                        
                        # add the binned attributes
                        categorical_attributes = {**categorical_attributes, **binned}

                        # Intialize a describer and a generator
                        describer = DataDescriber(category_threshold=threshold_value)
                        describer.describe_dataset_in_correlated_attribute_mode(f"{temp_files_dir}/temp.csv",
                                                                                epsilon=eps, 
                                                                                k=2,
                                                                                attribute_to_is_categorical=categorical_attributes,
                                                                                attribute_to_is_candidate_key=candidate_keys,
                                                                                seed=np.random.randint(1000000))
                        describer.save_dataset_description_to_file(f"{temp_files_dir}/privbayes_description.csv")

                        generator = DataGenerator()
                        generator.generate_dataset_in_correlated_attribute_mode(len(df),
                                                                                f"{temp_files_dir}/privbayes_description.csv")
                        generator.save_synthetic_data(f"{temp_files_dir}/privbayes_synth.csv")
                        privbayes_synth_data = pd.read_csv(f"{temp_files_dir}/privbayes_synth.csv")
                        
                        if self.slide_range:
                            privbayes_synth_data = self.slide_range_backward(privbayes_synth_data, range_transform)
                        
                        privbayes_synth_data.to_pickle(folder_name + 'privbayes_' + str(it) + '.pickle')
                        print(privbayes_synth_data.apply(lambda x: x.unique()))
                        print(privbayes_synth_data.apply(lambda x: len(x.unique())))

                    print('DONE: PrivBayes.')

