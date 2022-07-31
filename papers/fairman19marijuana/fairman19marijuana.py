from papers.meta_classes import Publication, Finding
import pandas as pd
import numpy as np
import os
import logging


class Fairman19Marijuana(Publication):
    DEFAULT_PAPER_ATTRIBUTES = {
        'length_pages': 16,
        'authors': ['Brian J Fairman', 'C Debra Furr-Holden', 'Renee M Johnson'],
        'journal': 'Prevention Science',
        'year': 2019,
        'current_citations': 39
    }
    DATAFRAME_COLUMNS = ['YEAR', 'CLASS', 'SEX', 'RACE', 'AGE_GROUP', 'AGE']
    INPUT_FILES = [
        'data/32722-0001-Data.tsv', 'data/23782-0001-Data.tsv', 'data/04596-0001-Data.tsv',
        'data/26701-0001-Data.tsv', 'data/29621-0001-Data.tsv', 'data/36361-0001-Data.tsv',
        'data/35509-0001-Data.tsv', 'data/04373-0001-Data.tsv', 'data/21240-0001-Data.tsv',
        'data/34481-0001-Data.tsv', 'data/34933-0001-Data.tsv'
    ]
    INPUT_FIELDS = [
        'NEWRACE', 'AGE', 'IRSEX', 'HERAGE', 'LSDAGE', 'PCPAGE', 'CRKAGE', 'ECSAGE', 'COCAGE', 'METHAGE', 'CIGAGE',
        'SNUFTRY', 'CHEWTRY', 'MTHAAGE', 'OXYCAGE', 'CIGTRY ', 'SEDAGE', 'STIMAGE', 'TRANAGE', 'CIGARTRY', 'INHAGE',
        'MJAGE', 'ANALAGE', 'BLNTAGE', 'ALCTRY'
    ]
    FILE_YEAR_MAP = {f: 2004 + i for i, f in enumerate(sorted([os.path.basename(f) for f in INPUT_FILES]))}
    CLASS_MAP = {
        'MJAGE': 'MARIJUANA',
        'ALCTRY': 'ALCOHOL',
        'CIGAGE': 'CIGARETTES',
        'NOUSAGE': 'NO_DRUG_USE',
        'CIGARTRY': 'OTHER_TABACCO',
        'SNUFTRY': 'OTHER_TABACCO',
        'CHEWTRY': 'OTHER_TABACCO',
        'HERAGE': 'OTHER_DRUGS',
        'LSDAGE': 'OTHER_DRUGS',
        'PCPAGE': 'OTHER_DRUGS',
        'CRKAGE': 'OTHER_DRUGS',
        'ECSAGE': 'OTHER_DRUGS',
        'COCAGE': 'OTHER_DRUGS',
        'METHAGE': 'OTHER_DRUGS',
        'MTHAAGE': 'OTHER_DRUGS',
        'OXYCAGE': 'OTHER_DRUGS',
        'SEDAGE': 'OTHER_DRUGS',
        'STIMAGE': 'OTHER_DRUGS',
        'TRANAGE': 'OTHER_DRUGS',
        'INHAGE': 'OTHER_DRUGS',
        'ANALAGE': 'OTHER_DRUGS',
        'BLNTAGE': 'OTHER_DRUGS'
    }
    AGE_GROUP_MAP = {
        12: '12-13',
        13: '12-13',
        14: '14-15',
        15: '14-15',
        16: '16-17',
        17: '16-17',
        18: '18-19',
        19: '18-19',
        20: '20-21',
        21: '20-21'
    }
    CLASSES_ = [
        'MJAGE', 'CIGAGE', 'ALCTRY', 'CIGARTRY', 'SNUFTRY',
        'CHEWTRY', 'HERAGE', 'LSDAGE', 'PCPAGE', 'CRKAGE',
        'ECSAGE', 'COCAGE', 'METHAGE', 'MTHAAGE', 'OXYCAGE',
        'SEDAGE', 'STIMAGE', 'TRANAGE', 'INHAGE', 'ANALAGE',
        'BLNTAGE'
    ]
    CLASSES = CLASSES_ + ['NOUSAGE']

    def __init__(self, dataframe=None, filename=None):
        if filename is not None:
            dataframe = pd.read_pickle(filename)
        elif dataframe is not None:
            dataframe = dataframe
        else:
            dataframe = self._recreate_dataframe()
        super().__init__(dataframe)
        self.FINDINGS = self.FINDINGS + [
            Finding(self.finding_5_1_5, description="finding_5_1", args={'class_name': 'MARIJUANA','expected_difference': .5},
                    text="""For each substance, the mean age of reported first use increased over the study period.
                    The mean age of first marijuana use increased ( 0.5 years) 
                    from 14.7 years in 2004 to 15.2 years in 2014;."""),
            Finding(self.finding_5_1_5, description="finding_5_2", args={'class_name': 'CIGARETTES', 'expected_difference': 1.4},
                    text='these numbers were comparable to those for age of first use of cigarettes (13.6 vs. 15.0; 1.4 years)'),
            Finding(self.finding_5_1_5, description="finding_5_3", args={'class_name':'ALCOHOL','expected_difference':.8},
                    text='alcohol (14.4 vs. 15.2;  0.8 years)'),
            Finding(self.finding_5_1_5, description="finding_5_4", args={'class_name': 'OTHER_TABACCO','expected_difference': .9},
                    text='other tobacco (14.8 vs. 15.7;  0.9 years)'),
            Finding(self.finding_5_1_5, description="finding_5_5", args={'class_name': 'OTHER_DRUGS','expected_difference': .6},
                    text='and other drug use (14.4 vs. 15.0;  0.6 years)')
        ]

    def _merge_input_files(self):
        dfs = []
        for file in self.INPUT_FILES:
            df_n1 = pd.read_csv(file, sep='\t', skipinitialspace=True, nrows=1)
            current_columns = []
            for field in self.INPUT_FIELDS:
                if field in df_n1.columns:
                    current_columns.append(field)
                elif f'{field}2' in df_n1.columns:
                    current_columns.append(f'{field}2')
                else:
                    logging.warning(f'field {field} not in {file}')
            df = pd.read_csv(file, sep='\t', skipinitialspace=True, usecols=current_columns)
            df['file_name'] = os.path.basename(file)
            dfs.append(df)
        return pd.concat(dfs)

    def _recreate_dataframe(self, filename='fairman19marijuana_dataframe.pickle'):
        df = self._merge_input_files()
        df['YEAR'] = df['file_name'].map(self.FILE_YEAR_MAP)  # infer year
        df[['MTHAAGE', 'BLNTAGE']] = df[['MTHAAGE', 'BLNTAGE']].fillna(10e5)  # fill in nan
        # data diff 296467 (real) - 297632 (paper) = -1165
        df = df[(df['AGE2'] < 11)]  # filter people < 22 yo
        df['ARGMINAGE'] = df[self.CLASSES_].values.argmin(axis=1)  # index of substance that was used first
        df['MINAGE'] = df[self.CLASSES_].values.min(axis=1)  # substance that was used first
        df['ARGMINAGE'] = np.where(df['MINAGE'] > 900, 21, df['ARGMINAGE'])  # values > 900 - no drug usage
        # mapping to good-looking values
        df['SEX'] = df['IRSEX'].map({1: 'Male', 2: 'Female'})
        df['AGE'] = df['AGE2'].map({i: i + 11 for i in range(1, 11)})
        df['RACE'] = df['NEWRACE2'].map(
            {1: 'White', 2: 'Black', 3: 'AI/AN', 4: 'NHOPI', 5: 'Asian', 6: 'Multi-racial', 7: 'Hispanic'})
        df['CLASS_NARROW'] = df['ARGMINAGE'].map(lambda x: self.CLASSES[x])
        df['CLASS'] = df['CLASS_NARROW'].map(self.CLASS_MAP)
        df['AGE_GROUP'] = df['AGE'].map(self.AGE_GROUP_MAP)
        df.reset_index(inplace=True, drop=True)
        df['SEX'] = df['SEX'].astype('category')
        df['RACE'] = df['RACE'].astype('category')
        df['AGE_GROUP'] = df['AGE_GROUP'].astype('category')
        df['CLASS'] = df['CLASS'].astype('category')
        df['YEAR'] = df['YEAR'].astype('category')
        df = df[self.DATAFRAME_COLUMNS]
        print(df.columns)
        df.to_pickle(filename)
        return df

    def finding_5_1_5(self, class_name, expected_difference):
        mean_first_marijuana_use_2004 = self.dataframe[
            (self.dataframe.CLASS == class_name) & (self.dataframe.YEAR == 2004)]['AGE'].mean()
        mean_first_marijuana_use_2014 = self.dataframe[
            (self.dataframe.CLASS == class_name) & (self.dataframe.YEAR == 2014)]['AGE'].mean()
        age_diff = np.round(mean_first_marijuana_use_2014 - mean_first_marijuana_use_2004, 2)
        return age_diff == expected_difference


if __name__ == '__main__':
    paper = Fairman19Marijuana(filename='fairman19marijuana_dataframe.pickle')
    for find in paper.FINDINGS:
        print(find.run())
