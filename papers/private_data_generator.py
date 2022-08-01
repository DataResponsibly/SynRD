import pandas as pd
import numpy as np
import os

from snsynth.mst import MSTSynthesizer
from snsynth.preprocessors import GeneralTransformer, BaseTransformer
from snsynth.pytorch import PytorchDPSynthesizer
from snsynth.pytorch.nn import PATECTGAN

from meta_classes import Publication

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
        "saw2018cross": "domains/saw2018cross-domain.json"
    }

    def __init__(self, publication):
        self.publication = publication
    
    def generate(self):
        df = self.publication.dataframe

        df_map = {
            self.publication.DEFAULT_PAPER_ATTRIBUTES['id'] : df
        }

        for pub_name, df in df_map.items():
            print('Generating: ' + pub_name)
            if not os.path.exists('private_data/' + str(pub_name)):
                os.mkdir('private_data/' + str(pub_name))
            for (eps, str_eps) in self.EPSILONS:
                if not os.path.exists('private_data/' + str(pub_name) + '/' + str_eps):
                    os.mkdir('private_data/' + str(pub_name) + '/' + str_eps)
                for it in range(self.ITERATIONS):
                    # Folder for deposit
                    folder_name = 'private_data/' + str(pub_name) + '/' + str_eps + '/'
                    
                    if not os.path.isfile(folder_name + 'mst_' + str(it) + '.pickle'):
                        # The MST Synthesis
                        mst = MSTSynthesizer(domain=pub_name, domains_dict=self.DOMAINS)
                        mst.fit(df)
                        sample_size = len(df)
                        mst_synth_data = mst.sample(sample_size)
                        mst_synth_data.to_pickle(folder_name + 'mst_' + str(it) + '.pickle')

                    if not os.path.isfile(folder_name + 'patectgan_' + str(it) + '.pickle'):
                        # The PATECTGAN Synthesis
                        preprocess_factor = 0.1
                        patectgan = PytorchDPSynthesizer(eps, 
                                                        PATECTGAN(preprocessor_eps=(preprocess_factor * eps)), 
                                                        preprocessor=None)
                        patectgan.fit(
                            df,
                            categorical_columns=list(df.columns),
                            transformer=BaseTransformer,
                        )
                        sample_size = len(df)
                        patectgan_synth_data = patectgan.sample(sample_size)
                        patectgan_synth_data.to_pickle(folder_name + 'patectgan_' + str(it) + '.pickle') 

            

