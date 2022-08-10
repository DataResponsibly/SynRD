from papers.meta_classes import Publication, Finding
from papers.meta_classes import NonReproducibleFindingException
from papers.file_utils import PathSearcher
import pandas as pd
import numpy as np
import os


class Fairman2019Marijuana(Publication):
    DEFAULT_PAPER_ATTRIBUTES = {
        'id': 'fairman2019marijuana',
        'length_pages': 16,
        'authors': ['Brian J Fairman', 'C Debra Furr-Holden', 'Renee M Johnson'],
        'journal': 'Prevention Science',
        'year': 2019,
        'current_citations': 39,
        'base_dataframe_pickle': 'fairman2019marijuana_dataframe.pickle'
    }
    DATAFRAME_COLUMNS = ['YEAR', 'CLASS', 'SEX', 'RACE', 'AGE', 'MINAGE']
    INPUT_FILES = [
        'data/32722-0001-Data.tsv', 'data/23782-0001-Data.tsv', 'data/04596-0001-Data.tsv',
        'data/26701-0001-Data.tsv', 'data/29621-0001-Data.tsv', 'data/36361-0001-Data.tsv',
        'data/35509-0001-Data.tsv', 'data/04373-0001-Data.tsv', 'data/21240-0001-Data.tsv',
        'data/34481-0001-Data.tsv', 'data/34933-0001-Data.tsv'
    ]
    INPUT_FIELDS = [
        'NEWRACE', 'AGE','IRSEX', 'USEACM', 'CIGTRY', 'ALCTRY', 'MJAGE', 'CIGARTRY', 'CHEWTRY', 'SNUFTRY', 'SLTTRY',
        'COCAGE', 'HALLAGE', 'HERAGE', 'INHAGE', 'ANALAGE', 'SEDAGE', 'STIMAGE', 'TRANAGE'
    ]
    FILE_YEAR_MAP = {f: i for i, f in enumerate(sorted([os.path.basename(f) for f in INPUT_FILES]))}
    YEAR_MAP = {2004 + i: i for i in range(len(FILE_YEAR_MAP))}
    AGE_MAP = {i: i + 12 for i in range(0, 10)}
    AGE_GROUP_MAP = {
        0: '12-13', 1: '12-13', 2: '14-15', 3: '14-15', 4: '16-17', 5: '16-17', 6: '18-19', 7: '18-19',
        8: '20-21', 9: '20-21'
    }
    SEX_MAP = {'Male': 0, 'Female': 1}
    RACE_MAP = {'White': 0, 'Black': 1, 'AI/AN': 2, 'NHOPI': 3, 'Asian': 4, 'Multi-racial': 5, 'Hispanic': 6}
    CLASSES = {
        'CIGTRY': 0, 'ALCTRY': 1, 'MJAGE': 2, 'CIGARTRY': 3, 'CHEWTRY': 3, 'SNUFTRY': 3, 'SLTTRY': 3, 'COCAGE': 4,
        'HALLAGE': 4, 'HERAGE': 4, 'INHAGE': 4, 'ANALAGE': 4, 'SEDAGE': 4, 'STIMAGE': 4, 'TRANAGE': 4, 'NOUSAGE': 5
    }
    CLASSES_PRETTY = {
        'CIGARETTES': 0, 'ALCOHOL': 1, 'MARIJUANA': 2, 'OTHER_TABACCO': 3, 'OTHER_DRUGS': 4, 'NOUSAGE': 5
    }
    USE_ACM_MAP = {
        1: CLASSES['ALCTRY'], 2: CLASSES['CIGTRY'], 3: CLASSES['MJAGE'],
        4: CLASSES['ALCTRY'], 5: CLASSES['CIGTRY'], 6: CLASSES['MJAGE'], 91: CLASSES['NOUSAGE'],
    }

    def __init__(self, dataframe=None, filename=None, path=None):
        if dataframe is None:
            if path is None:
                path = self.DEFAULT_PAPER_ATTRIBUTES['id']
            self.path_searcher = PathSearcher(path)
            if filename is None:
                filename = self.DEFAULT_PAPER_ATTRIBUTES['base_dataframe_pickle']
            try:
                dataframe = pd.read_pickle(self.path_searcher.get_path(filename))
            except FileNotFoundError:
                dataframe = self._recreate_dataframe()
        super().__init__(dataframe)
        self.FINDINGS = self.FINDINGS + [
            Finding(self.finding_5_1, description='finding_5_1', text="""
                For each substance, the mean age of reported first use increased over the study period.
                The mean age of first marijuana use increased ( 0.5 years)  from 14.7 years in 2004 to 15.2 years in 2014;.
                """),
            Finding(self.finding_5_2, description='finding_5_2', text="""
                these numbers were comparable to those for age of first use of cigarettes (13.6 vs. 15.0; 1.4 years)
                """),
            Finding(self.finding_5_3, description='finding_5_3', text="""
                alcohol (14.4 vs. 15.2;  0.8 years)
                """),
            Finding(self.finding_5_4, description='finding_5_4', text="""
                other tobacco (14.8 vs. 15.7;  0.9 years)
                """),
            Finding(self.finding_5_5, description='finding_5_5', text="""
                and other drug use (14.4 vs. 15.0;  0.6 years)
                """),
            Finding(self.finding_5_6, description='finding_5_6', text="""
                Aggregated across survey years, 5.8% of respondents reported that they initiated marijuana before other substances,
                compared to 29.8% for alcohol, 14.2% for cigarettes, 3.6% for other tobacco products, and 5.9% other drugs
                (these data are provided in online supplemental Table S1)
                """),
            Finding(self.finding_5_7, description='finding_5_7', text="""
                From 2004 to 2014, the proportion who had initiated marijuana before other substances increased
                from 4.4% to 8.0% (Figure 1),
                """),
            Finding(self.finding_5_8, description='finding_5_8', text="""
                declined for those having initiated cigarettes first (21.4% to 8.9%)
                """),
            Finding(self.finding_5_9, description='finding_5_9', text="""
                and increased in youth having abstained from substance use (35.5% to 46.3%)
                """),
            Finding(self.finding_5_10, description='finding_5_10', text="""
                Males were more likely than females to have initiated marijuana first (7.1%)
                """),
            Finding(self.finding_5_11, description='finding_5_11', text="""
                or other tobacco products first (5.7%),
                """),
            Finding(self.finding_5_12, description='finding_5_12', text="""
                whereas females were more likely than males to have initiated cigarettes (15.2%)
                """),
            Finding(self.finding_5_13, description='finding_5_13', text="""
                or alcohol first (32.0%).
                """),
            Finding(self.finding_5_14, description='finding_5_14', text="""
                Considering age, a small proportion of 12–13-year-olds (0.6%) reported initiating marijuana before other substances,
                but by ages 18–19 and 20–21-years this proportion increased to 9.1%
                """),
            Finding(self.finding_6_1, description='finding_6_1', text="""
                and 9.4%, respectively.
                """),
            Finding(self.finding_6_2, description='finding_6_2', text="""
                American Indian/Alaskan Native (AI/AN) (11.8%) and Black youth (9.4%) had the highest proportion of initiating
                marijuana first;
                """),
            Finding(self.finding_6_3, description='finding_6_3', text="""
                White (4.6%) and Asian youth (2.5% had the lowest).
                """),
        ]

    def _merge_input_files(self):
        dfs = []
        for file in self.INPUT_FILES:
            df_n1 = pd.read_csv(file, sep='\t', skipinitialspace=True, nrows=1)
            current_columns = []
            for field in self.INPUT_FIELDS:
                field = field.strip()
                if field in df_n1.columns:
                    current_columns.append(field)
                elif f'{field}2' in df_n1.columns:
                    current_columns.append(f'{field}2')
                else:
                    print(f'field {field} not in {file}')
            df = pd.read_csv(file, sep='\t', skipinitialspace=True, usecols=current_columns)
            df['file_name'] = os.path.basename(file)
            dfs.append(df)
        return pd.concat(dfs)

    def _recreate_dataframe(self, filename='fairman2019marijuana_dataframe.pickle'):
        main_df = self._merge_input_files()
        main_df = main_df[(main_df['AGE2'] < 11)]  # filter people < 22 yo
        drugs_classes = list(self.CLASSES)[:-1]
        main_df['MINAGE'] = main_df[drugs_classes].values.min(axis=1)
        main_df['MINAGE'] = np.where(main_df['MINAGE'] > 900, 999, main_df['MINAGE'])
        main_df['MINAGE_CLASS'] = np.where(main_df['MINAGE'] > 900, self.CLASSES['NOUSAGE'], None)
        main_df['CLASSES_LIST'] = np.where(main_df['MINAGE'] > 900, str(self.CLASSES['NOUSAGE']), None)
        main_df = main_df[~((main_df.MINAGE_CLASS == self.CLASSES['NOUSAGE']) &
                            (main_df.USEACM == 99))]  # remove where unknown class
        main_df['YEAR'] = main_df['file_name'].map(self.FILE_YEAR_MAP)  # infer year
        # make all vars to range from 0 to max(var) - required for mst
        main_df['SEX'] = main_df['IRSEX'] - 1
        main_df['RACE'] = main_df['NEWRACE2'] - 1
        main_df['AGE'] = main_df['AGE2'] - 1
        main_df.reset_index(inplace=True, drop=True)
        for i, row in main_df.iterrows():
            if row['MINAGE'] > 900:  # used smth
                continue
            several_substances = sorted(
                row[drugs_classes][row[drugs_classes].apply(lambda x: x == row['MINAGE'])].index.values)
            several_substances_mapped = sorted(list(set([self.CLASSES[s] for s in several_substances])))
            main_df.at[i, 'CLASSES_LIST'] = '/'.join(map(str, several_substances_mapped))
            if len(several_substances_mapped) == 1:
                main_df.at[i, 'MINAGE_CLASS'] = several_substances_mapped[0]
            else:
                main_df.at[i, 'MINAGE_CLASS'] = self.USE_ACM_MAP.get(row['USEACM']) or np.random.choice(several_substances_mapped)
        main_df['CLASS'] = main_df['MINAGE_CLASS']
        main_df['MINAGE'] = np.where(main_df['MINAGE'] > 900, 0, main_df['MINAGE'])
        main_df.reset_index(inplace=True, drop=True)
        df = main_df[self.DATAFRAME_COLUMNS]
        df = df.astype(np.int32)  # all features categorical but numerically encoded
        print(df.shape)
        print(df.columns)
        print(df.dtypes)
        df.to_pickle(filename)
        return df

    def _get_mean_minage_class_year(self, class_name: str, year: int):
        return self.dataframe[(self.dataframe.CLASS == self.CLASSES_PRETTY[class_name]) & (
                self.dataframe.YEAR == self.YEAR_MAP[year])]['MINAGE'].astype(np.int32).mean()

    def finding_5_1(self):
        """
        For each substance, the mean age of reported first use increased over the study period.
        The mean age of first marijuana use increased ( 0.5 years)  from 14.7 years in 2004 to 15.2 years in 2014;.
        """
        mean_first_marijuana_use_2004 = self._get_mean_minage_class_year('MARIJUANA', 2004)
        mean_first_marijuana_use_2014 = self._get_mean_minage_class_year('MARIJUANA', 2014)
        age_diff = np.abs(mean_first_marijuana_use_2014 - mean_first_marijuana_use_2004)
        findings = [mean_first_marijuana_use_2004, mean_first_marijuana_use_2014]
        soft_finding = mean_first_marijuana_use_2014 > mean_first_marijuana_use_2004
        hard_findings = [np.allclose(age_diff, 0.5, atol=10e-2)]
        return findings, soft_finding, hard_findings

    def finding_5_2(self):
        """
        these numbers were comparable to those for age of first use of cigarettes (13.6 vs. 15.0; 1.4 years)
        """
        mean_age_first_use_2004 = self._get_mean_minage_class_year('CIGARETTES', 2004)
        mean_age_first_use_2014 = self._get_mean_minage_class_year('CIGARETTES', 2014)
        age_diff = np.round(mean_age_first_use_2014 - mean_age_first_use_2004, 2)
        findings = [mean_age_first_use_2004, mean_age_first_use_2014]
        soft_finding = mean_age_first_use_2014 > mean_age_first_use_2004
        hard_findings = [np.allclose(age_diff, 1.4, atol=10e-2)]
        return findings, soft_finding, hard_findings

    def finding_5_3(self):
        """
        alcohol (14.4 vs. 15.2;  0.8 years)
        """
        mean_age_first_use_2004 = self._get_mean_minage_class_year('ALCOHOL', 2004)
        mean_age_first_use_2014 = self._get_mean_minage_class_year('ALCOHOL', 2014)
        age_diff = np.round(mean_age_first_use_2014 - mean_age_first_use_2004, 1)
        findings = [mean_age_first_use_2004, mean_age_first_use_2014]
        soft_finding = mean_age_first_use_2014 > mean_age_first_use_2004
        hard_findings = [np.allclose(age_diff, 0.8, atol=10e-2)]
        return findings, soft_finding, hard_findings

    def finding_5_4(self):
        """
        other tobacco (14.8 vs. 15.7;  0.9 years)
        """
        mean_age_first_use_2004 = self._get_mean_minage_class_year('OTHER_TABACCO', 2004)
        mean_age_first_use_2014 = self._get_mean_minage_class_year('OTHER_TABACCO', 2014)
        age_diff = np.round(mean_age_first_use_2014 - mean_age_first_use_2004, 1)
        findings = [mean_age_first_use_2004, mean_age_first_use_2014]
        soft_finding = mean_age_first_use_2014 > mean_age_first_use_2004
        hard_findings = [np.allclose(age_diff, 0.9, atol=10e-2)]
        return findings, soft_finding, hard_findings

    def finding_5_5(self):
        """
        and other drug use (14.4 vs. 15.0;  0.6 years)
        """
        mean_age_first_use_2004 = self._get_mean_minage_class_year('OTHER_DRUGS', 2004)
        mean_age_first_use_2014 = self._get_mean_minage_class_year('OTHER_DRUGS', 2014)
        age_diff = np.round(mean_age_first_use_2014 - mean_age_first_use_2004, 1)
        findings = [mean_age_first_use_2004, mean_age_first_use_2014]
        soft_finding = mean_age_first_use_2014 > mean_age_first_use_2004
        hard_findings = [np.allclose(age_diff, 0.6, atol=10e-2)]
        return findings, soft_finding, hard_findings

    def finding_5_6(self):
        """
        Aggregated across survey years, 5.8% of respondents reported that they initiated marijuana before other substances,
        compared to 29.8% for alcohol, 14.2% for cigarettes, 3.6% for other tobacco products, and 5.9% other drugs
        (these data are provided in online supplemental Table S1)
        """
        table = self.dataframe.CLASS.value_counts() / self.dataframe.shape[0]
        marijuana_ratio = table[self.CLASSES_PRETTY['MARIJUANA']] * 100
        alcohol_ratio = table[self.CLASSES_PRETTY['ALCOHOL']] * 100
        cigarettes_ratio = table[self.CLASSES_PRETTY['CIGARETTES']] * 100
        other_tobacco_ratio = table[self.CLASSES_PRETTY['OTHER_TABACCO']] * 100
        other_drugs_ratio = table[self.CLASSES_PRETTY['OTHER_DRUGS']] * 100
        findings = [marijuana_ratio, alcohol_ratio, cigarettes_ratio, other_drugs_ratio, other_tobacco_ratio]
        soft_finding = (marijuana_ratio < alcohol_ratio) & (marijuana_ratio < cigarettes_ratio) & \
                       (marijuana_ratio > other_tobacco_ratio)
        hard_findings = [np.allclose(marijuana_ratio, 5.8, atol=10e-2), np.allclose(alcohol_ratio, 29.8, atol=10e-2),
            np.allclose(cigarettes_ratio, 14.2, atol=10e-2), np.allclose(other_drugs_ratio, 5.9, atol=10e-2),
            np.allclose(other_tobacco_ratio, 3.6, atol=10e-2)]
        return findings, soft_finding, hard_findings

    def _get_prop_class_year(self, class_name: str, year: int):
        return self.dataframe[(self.dataframe.CLASS == self.CLASSES_PRETTY[class_name]) & (
                self.dataframe.YEAR == self.YEAR_MAP[year])].shape[0] * 100 / \
               self.dataframe[ (self.dataframe.YEAR == self.YEAR_MAP[year])].shape[0]

    def finding_5_7(self):
        """
        From 2004 to 2014, the proportion who had initiated marijuana before other substances increased
        from 4.4% to 8.0% (Figure 1),
        """
        marijuana_prop_2004 = self._get_prop_class_year('MARIJUANA', 2004)
        marijuana_prop_2014 = self._get_prop_class_year('MARIJUANA', 2014)
        findings = [marijuana_prop_2004, marijuana_prop_2014]
        soft_finding = marijuana_prop_2004 < marijuana_prop_2014
        hard_findings = [np.allclose(marijuana_prop_2004, 4.4, atol=10e-2), np.allclose(marijuana_prop_2014, 8.0, atol=10e-2)]
        return findings, soft_finding, hard_findings

    def finding_5_8(self):
        """
        declined for those having initiated cigarettes first (21.4% to 8.9%)
        """
        cig_prop_2004 = self._get_prop_class_year('CIGARETTES', 2004)
        cig_prop_2014 = self._get_prop_class_year('CIGARETTES', 2014)
        findings = [cig_prop_2004, cig_prop_2014]
        soft_finding = cig_prop_2004 > cig_prop_2014
        hard_findings = [np.allclose(cig_prop_2004, 21.4, atol=10e-2), np.allclose(cig_prop_2014, 8.9, atol=10e-2)]
        return findings, soft_finding, hard_findings

    def finding_5_9(self):
        """
        and increased in youth having abstained from substance use (35.5% to 46.3%)
        """
        no_usage_prop_2004 = self._get_prop_class_year('NOUSAGE', 2004)
        no_usage_prop_2014 = self._get_prop_class_year('NOUSAGE', 2014)
        findings = [no_usage_prop_2004, no_usage_prop_2014]
        soft_finding = no_usage_prop_2004 < no_usage_prop_2014
        hard_findings = [np.allclose(no_usage_prop_2004, 35.5, atol=10e-2), np.allclose(no_usage_prop_2014, 46.3, atol=10e-2)]
        return findings, soft_finding, hard_findings

    def table_s1(self, feature):
        if 'AGE_GROUP' not in self.dataframe.columns:
            self.dataframe['AGE_GROUP'] = self.dataframe['AGE'].map(self.AGE_GROUP_MAP)
        return self.dataframe[['CLASS', feature]].value_counts() / self.dataframe[[feature]].value_counts()

    def finding_5_10(self):
        """
        Males were more likely than females to have initiated marijuana first (7.1%)
        """
        table = self.table_s1(feature='SEX')
        male_marijuana_ratio = table[self.SEX_MAP['Male'], self.CLASSES_PRETTY['MARIJUANA']] * 100
        female_marijuana_ratio = table[self.SEX_MAP['Female'], self.CLASSES_PRETTY['MARIJUANA']] * 100
        findings = [male_marijuana_ratio, female_marijuana_ratio]
        soft_finding = male_marijuana_ratio > female_marijuana_ratio
        hard_findings = [np.allclose(male_marijuana_ratio, 7.1, atol=10e-2)]
        return findings, soft_finding, hard_findings

    def finding_5_11(self):
        """
        or other tobacco products first (5.7%),
        """
        table = self.table_s1(feature='SEX')
        male_other_tobacco_ratio = table[self.SEX_MAP['Male'], self.CLASSES_PRETTY['OTHER_TABACCO']] * 100
        female_other_tobacco_ratio = table[self.SEX_MAP['Female'], self.CLASSES_PRETTY['OTHER_TABACCO']] * 100
        findings = [female_other_tobacco_ratio, male_other_tobacco_ratio]
        soft_finding = male_other_tobacco_ratio > female_other_tobacco_ratio
        hard_findings = [np.allclose(male_other_tobacco_ratio,  5.7, atol=10e-2)]
        return findings, soft_finding, hard_findings

    def finding_5_12(self):
        """
        whereas females were more likely than males to have initiated cigarettes (15.2%)
        """
        table = self.table_s1(feature='SEX')
        male_cigarettes_ratio = table[self.SEX_MAP['Male'], self.CLASSES_PRETTY['CIGARETTES']] * 100
        female_cigarettes_ratio = table[self.SEX_MAP['Female'], self.CLASSES_PRETTY['CIGARETTES']] * 100
        findings = [female_cigarettes_ratio, male_cigarettes_ratio]
        soft_finding = male_cigarettes_ratio < female_cigarettes_ratio
        hard_findings = [np.allclose(female_cigarettes_ratio, 15.2, atol=10e-2)]
        return findings, soft_finding, hard_findings

    def finding_5_13(self):
        """
        or alcohol first (32.0%)
        """
        table = self.table_s1(feature='SEX')
        male_alcohol_ratio = table[self.SEX_MAP['Male'], self.CLASSES_PRETTY['ALCOHOL']] * 100
        female_alcohol_ratio = table[self.SEX_MAP['Female'], self.CLASSES_PRETTY['ALCOHOL']] * 100
        findings = [male_alcohol_ratio, female_alcohol_ratio]
        soft_finding = male_alcohol_ratio < female_alcohol_ratio
        hard_findings = [np.allclose(female_alcohol_ratio,  32.0, atol=10e-2)]
        return findings, soft_finding, hard_findings

    def finding_5_14(self):
        """
        Considering age, a small proportion of 12–13-year-olds (0.6%) reported initiating marijuana before other substances,
        but by ages 18–19 and 20–21-years this proportion increased to 9.1%
        """
        table = self.table_s1(feature='AGE_GROUP')
        youngest_marijuana_ratio = table['12-13', self.CLASSES_PRETTY['MARIJUANA']] * 100
        high_school_grads_marijuana_ratio = table['18-19', self.CLASSES_PRETTY['MARIJUANA']] * 100
        soft_finding = high_school_grads_marijuana_ratio > youngest_marijuana_ratio
        hard_findings = [np.allclose(youngest_marijuana_ratio, 0.6, atol=10e-2),
                         np.allclose(high_school_grads_marijuana_ratio, 9.1, atol=10e-2)]
        return [youngest_marijuana_ratio, high_school_grads_marijuana_ratio], soft_finding, hard_findings

    def finding_6_1(self):
        """
        and 9.4%, respectively.
        """
        table = self.table_s1(feature='AGE_GROUP')
        youngest_marijuana_ratio = table['12-13', self.CLASSES_PRETTY['MARIJUANA']] * 100
        oldest_marijuana_ratio = table['20-21', self.CLASSES_PRETTY['MARIJUANA']] * 100
        soft_finding = oldest_marijuana_ratio > youngest_marijuana_ratio
        hard_findings = [np.allclose(oldest_marijuana_ratio, 9.4, atol=10e-2)]
        return [youngest_marijuana_ratio, oldest_marijuana_ratio], soft_finding, hard_findings

    def finding_6_2(self):
        """
        American Indian/Alaskan Native (AI/AN) (11.8%) and Black youth (9.4%) had the highest proportion of initiating
        marijuana first;
        """
        table = self.table_s1(feature='RACE')
        aian_marijuana_ratio = table[self.RACE_MAP['AI/AN'], self.CLASSES_PRETTY['MARIJUANA']] * 100
        black_marijuana_ratio = table[self.RACE_MAP['Black'], self.CLASSES_PRETTY['MARIJUANA']] * 100
        all_marijuana_sorted = sorted(
            [table[race, self.CLASSES_PRETTY['MARIJUANA']] * 100 for race in self.dataframe.RACE.unique()])
        soft_finding = sorted(all_marijuana_sorted[-2:]) == sorted([aian_marijuana_ratio, black_marijuana_ratio])
        hard_findings = [np.allclose(aian_marijuana_ratio, 11.8, atol=10e-2),
                         np.allclose(black_marijuana_ratio, 9.4, atol=10e-2)]
        return [aian_marijuana_ratio, black_marijuana_ratio], soft_finding, hard_findings

    def finding_6_3(self):
        """
        White (4.6%) and Asian youth (2.5% had the lowest).
        """
        table = self.table_s1(feature='RACE')
        white_marijuana_ratio = table[self.RACE_MAP['White'], self.CLASSES_PRETTY['MARIJUANA']] * 100
        asian_marijuana_ratio = table[self.RACE_MAP['Asian'], self.CLASSES_PRETTY['MARIJUANA']] * 100
        all_marijuana_sorted = sorted(
            [table[race, self.CLASSES_PRETTY['MARIJUANA']] * 100 for race in self.dataframe.RACE.unique()])
        soft_finding = sorted(all_marijuana_sorted[:2]) == sorted([white_marijuana_ratio, asian_marijuana_ratio])
        hard_findings = [np.allclose(white_marijuana_ratio, 9.4, atol=10e-2),
                         np.allclose(asian_marijuana_ratio, 9.4, atol=10e-2)]
        return [white_marijuana_ratio, asian_marijuana_ratio], soft_finding, hard_findings

    def finding_6_4(self):
        """
        As shown in Table 1, males were more likely than females to have initiated marijuana first in comparison to
        those not using drugs (aRRR = 1.69), those initiating cigarettes first (aRRR = 1.79), or
        those initiating alcohol first (aRRR = 1.83)
        """
        raise NonReproducibleFindingException

    def finding_6_5(self):
        """
        Likewise, the likelihood of initiating marijuana first relative to no drug use (aRRR = 1.69) or alcohol first
        (aRRR = 1.06) increased with age, but not relative to initiating cigarettes first.
        """
        raise NonReproducibleFindingException

    def finding_6_6(self):
        """
        Compared to Whites, AI/AN youth were 3.7 times more likely to have initiated marijuana first relative to no drug
        use, and were 5.0 times more likely to have initiated marijuana first relative to alcohol
        """
        raise NonReproducibleFindingException

    def finding_6_7(self):
        """
        Notably, Black youth were the most likely to have initiated marijuana first compared to cigarettes (aRRR = 2.74).
        """
        raise NonReproducibleFindingException

    def finding_6_8(self):
        """
        To a lesser extent, Hispanic, Native Hawaiian/Other Pacific Islander (NHOPI), and multiracial youth also had a
        higher likelihood of initiating marijuana before other substances compared to Whites.
        """
        raise NonReproducibleFindingException

    def finding_6_9(self):
        """
        By contrast, Asian youth were less likely to have initiated marijuana first relative to no drug use (aRRR = 0.30)
        or alcohol first (aRRR = 0.59).
        """
        raise NonReproducibleFindingException

    def finding_6_10(self):
        """
        Thus, White and Asian youth were more likely to have initiated cigarettes or alcohol first before other
        substances compared to other racial/ethnic groups.
        """
        raise NonReproducibleFindingException

    def finding_6_11(self):
        """
        However, there was less variation by race/ethnicity among older age groups. For example, 20–21-yearold Black
        youth had a similar likelihood of initiating marijuana first relative to Whites, but 15–16-year-old Black youth
        had almost twice the likelihood (aRRR = 1.9).
        """
        raise NonReproducibleFindingException

    def finding_6_12(self):
        """
        We found no subgroup interactions by sex (i.e., age x sex or race/ethnicity x sex)
        """
        raise NonReproducibleFindingException

    def finding_6_13(self):
        """
        Generally, those who started with a particular substance were the most like to have prevalent problematic use
        of that substance. For example, those who initiated marijuana before other substances were more likely currently
        smoke marijuana heavily and have CUD. Those who initiated alcohol before other substances were the most likely to
        experience prevalent AUD, and those who initiated cigarettes first were the most likely to experience prevalent ND.
        """
        raise NonReproducibleFindingException

    def finding_6_14(self):
        """
        However, it is worth noting that those who initiated marijuana first were no less likely, statistically, to
        have prevalent ND as compared to those who initiated cigarettes first.
        """
        raise NonReproducibleFindingException

    def finding_6_15(self):
        """
        Finally, youth who initiated cigarettes or other tobacco products before other substances were less likely than
        those starting with alcohol or marijuana to have used other drugs, such as cocaine, heroin, inhalants, and
        non-medical prescription drugs.
        """
        raise NonReproducibleFindingException

    def finding_7_1(self):
        """
        We found that, in 2014, 8% of US youths aged 12–21-years reported that marijuana was the first drug they used;
        this percentage has almost doubled since 2004.
        """
        raise NonReproducibleFindingException


if __name__ == '__main__':
    paper = Fairman2019Marijuana()
    for find in paper.FINDINGS:
        print(find.description, find.run())
