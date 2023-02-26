import os
import requests
from tqdm import tqdm
import pandas as pd


class DataRetriever():
    MAPPINGS = {
        'saw2018cross': 
            [('HSLS09.tsv', 'https://dataverse.harvard.edu/api/access/datafile/6947112')],
        'fairman2019marijuana_processed': 
            [('fairman2019marijuana.tsv', 'https://dataverse.harvard.edu/api/access/datafile/6948439')],
        'saw2018cross_processed': 
            [('saw2018cross.tsv', 'https://dataverse.harvard.edu/api/access/datafile/6948440')],
        'fruiht2018naturally_processed': 
            [('fruiht2018naturally.tsv', 'https://dataverse.harvard.edu/api/access/datafile/6948437')],
        'iverson22football_processed': 
            [('iverson22football.tsv', 'https://dataverse.harvard.edu/api/access/datafile/6948441')],
        'jeong2021math_processed': 
            [('jeong2021math.tsv', 'https://dataverse.harvard.edu/api/access/datafile/6948438')],
        'lee2021ability_processed': 
            [('lee2021ability.tsv', 'https://dataverse.harvard.edu/api/access/datafile/6948436')],
        'pierce2019who_processed': 
            [('pierce2019who.tsv','https://dataverse.harvard.edu/api/access/datafile/6959655')],
        'assari2019baseline_processed': 
            [('assari2019baseline.tsv','https://dataverse.harvard.edu/api/access/datafile/6959652')]
    }

    def __init__(self, papers, preprocessed=True):
        checked_papers = []
        for p in papers:
            if not isinstance(p, str):
                p = str(p.__name__).lower()
            if preprocessed:
                p += '_processed'
            if p not in self.MAPPINGS.keys():
                raise ValueError(str(p) + ' data not in DataRetriever')
            checked_papers.append(p)

        self.papers = checked_papers

    def retrieve_necessary_data(self, return_df_map=True):
        if not os.path.exists('data'):
            os.makedirs('data')
        for paper in self.papers:
            for i, (name, f) in enumerate(self.MAPPINGS[paper]):
                # check if file exists in data folder
                if os.path.exists('data/' + name):
                    continue
                # if not, download file from url
                response = requests.get(f, stream=True)
                total_size_in_bytes= int(response.headers.get('content-length', 0))
                block_size = 1024 
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open('data/' + name, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
        if return_df_map:
            df_map = {}
            for paper in self.papers:
                for i, (name, f) in enumerate(self.MAPPINGS[paper]):
                    # load tsv from data folder into df
                    df = pd.read_csv('data/' + name, sep='\t')
                    # map df to paper name
                    df_map[paper] = df
            return df_map