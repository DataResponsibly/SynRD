from meta_classes import Publication, Finding

import pandas as pd
import numpy as np

from itertools import chain

class Saw2018Cross(Publication):
    """
    A class wrapper for all publication classes, for shared functionality.
    """
    DEFAULT_PAPER_ATTRIBUTES = {
        'length_pages': 8,
        'authors': ['Guan Saw', 'Chi-Ning Chang', 'Hsun-Yu Chan'],
        'journal': 'Educational Researcher',
        'year': 2018,
        'current_citations': 67
    }

    DATAFRAME_COLUMNS = ['sex',
                         'race',
                         'SES',
                         'HigherSES',
                         'X1STU30OCC_STEM1',
                         'X2STU30OCC_STEM1',
                         'ninth_grade_aspirations',
                         'eleventh_grade_aspirations',
                         'stem_career_aspirations']

    RACE_MAP = {
        0: "White",
        1: "Black",
        2: "Hispanic",
        3: "Asian",
        4: "Multiracial",
        5: "Other"
    }

    SEX_MAP = {
        0: "boys",
        1: "girls"
    }

    HSES_MAP = {
        0: "lower-SES",
        1: "higher-SES"
    }

    ASPIRATION_MAP = {
        0: "No",
        1: "Yes"
    }

    FILENAME = 'saw2018cross'

    FIGURE_2_REINDEX = [
                    (      'White', 'higher-SES',  'boys', 'Yes'),
                    (      'White',  'lower-SES',  'boys', 'Yes'),
                    (      'White', 'higher-SES', 'girls', 'Yes'),
                    (      'White',  'lower-SES', 'girls', 'Yes'),
                    (      'Black', 'higher-SES',  'boys', 'Yes'),
                    (      'Black',  'lower-SES',  'boys', 'Yes'),
                    (      'Black', 'higher-SES', 'girls', 'Yes'),
                    (      'Black',  'lower-SES', 'girls', 'Yes'),
                    (   'Hispanic', 'higher-SES',  'boys', 'Yes'),
                    (   'Hispanic',  'lower-SES',  'boys', 'Yes'),
                    (   'Hispanic', 'higher-SES', 'girls', 'Yes'),
                    (   'Hispanic',  'lower-SES', 'girls', 'Yes'),
                    (      'Asian', 'higher-SES',  'boys', 'Yes'),
                    (      'Asian',  'lower-SES',  'boys', 'Yes'),
                    (      'Asian', 'higher-SES', 'girls', 'Yes'),
                    (      'Asian',  'lower-SES', 'girls', 'Yes'),
                    ('Multiracial', 'higher-SES',  'boys', 'Yes'),
                    ('Multiracial',  'lower-SES',  'boys', 'Yes'),
                    ('Multiracial', 'higher-SES', 'girls', 'Yes'),
                    ('Multiracial',  'lower-SES', 'girls', 'Yes')
    ]

    TABLE_B2_REINDEX = [
                    (      'White', 'higher-SES',  'boys', 'n'),
                    (      'White', 'higher-SES',  'boys', 'Yes'),
                    (      'White',  'lower-SES',  'boys', 'n'),
                    (      'White',  'lower-SES',  'boys', 'Yes'),
                    (      'White', 'higher-SES', 'girls', 'n'),
                    (      'White', 'higher-SES', 'girls', 'Yes'),
                    (      'White',  'lower-SES', 'girls', 'n'),
                    (      'White',  'lower-SES', 'girls', 'Yes'),
                    (      'Black', 'higher-SES',  'boys', 'n'),
                    (      'Black', 'higher-SES',  'boys', 'Yes'),
                    (      'Black',  'lower-SES',  'boys', 'n'),
                    (      'Black',  'lower-SES',  'boys', 'Yes'),
                    (      'Black', 'higher-SES', 'girls', 'n'),
                    (      'Black', 'higher-SES', 'girls', 'Yes'),
                    (      'Black',  'lower-SES', 'girls', 'n'),
                    (      'Black',  'lower-SES', 'girls', 'Yes'),
                    (   'Hispanic', 'higher-SES',  'boys', 'n'),
                    (   'Hispanic', 'higher-SES',  'boys', 'Yes'),
                    (   'Hispanic',  'lower-SES',  'boys', 'n'),
                    (   'Hispanic',  'lower-SES',  'boys', 'Yes'),
                    (   'Hispanic', 'higher-SES', 'girls', 'n'),
                    (   'Hispanic', 'higher-SES', 'girls', 'Yes'),
                    (   'Hispanic',  'lower-SES', 'girls', 'n'),
                    (   'Hispanic',  'lower-SES', 'girls', 'Yes'),
                    (      'Asian', 'higher-SES',  'boys', 'n'),
                    (      'Asian', 'higher-SES',  'boys', 'Yes'),
                    (      'Asian',  'lower-SES',  'boys', 'n'),
                    (      'Asian',  'lower-SES',  'boys', 'Yes'),
                    (      'Asian', 'higher-SES', 'girls', 'n'),
                    (      'Asian', 'higher-SES', 'girls', 'Yes'),
                    (      'Asian',  'lower-SES', 'girls', 'n'),
                    (      'Asian',  'lower-SES', 'girls', 'Yes'),
                    ('Multiracial', 'higher-SES',  'boys', 'n'),
                    ('Multiracial', 'higher-SES',  'boys', 'Yes'),
                    ('Multiracial',  'lower-SES',  'boys', 'n'),
                    ('Multiracial',  'lower-SES',  'boys', 'Yes'),
                    ('Multiracial', 'higher-SES', 'girls', 'n'),
                    ('Multiracial', 'higher-SES', 'girls', 'Yes'),
                    ('Multiracial',  'lower-SES', 'girls', 'n'),
                    ('Multiracial',  'lower-SES', 'girls', 'Yes'),
                    (      'Other', 'higher-SES',  'boys', 'n'),
                    (      'Other', 'higher-SES',  'boys', 'Yes'),
                    (      'Other',  'lower-SES',  'boys', 'n'),
                    (      'Other',  'lower-SES',  'boys', 'Yes'),
                    (      'Other', 'higher-SES', 'girls', 'n'),
                    (      'Other', 'higher-SES', 'girls', 'Yes'),
                    (      'Other',  'lower-SES', 'girls', 'n'),
                    (      'Other',  'lower-SES', 'girls', 'Yes')
    ]

    def __init__(self, dataframe=None, filename=None):
        if filename is not None:
            self.dataframe = pd.read_pickle(filename)
        elif dataframe is not None:
            self.dataframe = dataframe
        else:
            self.dataframe = self._recreate_dataframe()

        self.FINDINGS = self.FINDINGS + [
            Finding(self.table_b2, description="table_b2"),
            Finding(self.figure_2, description="figure_2")
        ]
        

    def _recreate_dataframe(self, filename='saw2018cross_dataframe.pickle'):
        student_survey = pd.read_csv('saw2018cross/data/36423-0002-Data.tsv',sep='\t')

        filter1 = student_survey[student_survey['X2UNIV2A'] == 1]
        filter1 = filter1[(filter1['S1GRD0809'] != 3) & \
                 (filter1['S1GRD0809'] != 4) & (filter1['S1GRD0809'] > 0)]

        # SHIFT NEGATIVE VALS FOR X1STU30OCC_STEM1 and X2STU30OCC_STEM1 up to positive
        # vals
        filter1['X1STU30OCC_STEM1'] = filter1['X1STU30OCC_STEM1'].replace({-9 : 7})
        filter1['X2STU30OCC_STEM1'] = filter1['X2STU30OCC_STEM1'].replace({-9 : 7})
        filter1['X2STU30OCC_STEM1'] = filter1['X2STU30OCC_STEM1'].replace({-8 : 8})
        
        filter1['sex'] = filter1['X1SEX']
        filter1.loc[filter1["sex"] == 1, "sex"] = 0
        filter1.loc[filter1["sex"] == 2, "sex"] = 1
        filter1['sex'] = filter1['sex'].astype(np.int8)

        filter1['race'] = filter1['X1RACE']
        filter1.loc[filter1["race"] == 8, "race"] = 10
        filter1.loc[filter1["race"] == 3, "race"] = 11
        filter1.loc[filter1["race"] == 4, "race"] = 12
        filter1.loc[filter1["race"] == 5, "race"] = 12
        filter1.loc[filter1["race"] == 2, "race"] = 13
        filter1.loc[filter1["race"] == 6, "race"] = 14
        filter1.loc[filter1["race"] == 7, "race"] = 15
        filter1.loc[filter1["race"] == 1, "race"] = 15
        filter1['race'] = filter1['race'] - 10
        filter1['race'] = filter1['race'].astype(np.int8)

        filter1['SES'] = pd.qcut(filter1['X1SES_U'],4,labels=np.arange(4) + 1)
        filter1['SES'] = filter1['SES'].astype(np.int8)
        filter1['HigherSES'] = (filter1['SES'] > 2).astype(int)
        filter1['HigherSES'] = filter1['HigherSES'].astype(np.int8)

        filter1['ninth_grade_aspirations'] = (filter1['X1STU30OCC_STEM1'] == 1).astype(np.int8)
        filter1['eleventh_grade_aspirations'] = (filter1['X2STU30OCC_STEM1'] == 1).astype(np.int8)
        filter1['stem_career_aspirations'] = ((filter1['X1STU30OCC_STEM1'] == 1) | 
                                              (filter1['X2STU30OCC_STEM1'] == 1)).astype(np.int8)

        filter1 = filter1[self.DATAFRAME_COLUMNS]
        print(filter1.columns)
        filter1.to_pickle(filename)
        return filter1

    def _find_ninth_grade(self):
        """
        This produces the set of ninth graders who reported on their STEM aspirations
        validly in ninth grade
        """
        ninth_grade_set =  self.dataframe[self.dataframe['X1STU30OCC_STEM1'].isin([0,1,2,3,4,5,6])]
        ninth_grade_stem =  ninth_grade_set[ninth_grade_set['X1STU30OCC_STEM1'].isin([1,2,3])]
        ninth_grade_non_stem =  ninth_grade_set[~ninth_grade_set['X1STU30OCC_STEM1'].isin([1,2,3])]
        return ninth_grade_set, ninth_grade_stem, ninth_grade_non_stem

    def _find_eleventh_grade(self):
        """
        This produces the set of eleventh graders who reported on their STEM aspirations
        validly in eleventh grade
        """
        eleventh_grade_set =  self.dataframe[self.dataframe['X2STU30OCC_STEM1'].isin([0,1,2,3,4,5,6])]
        eleventh_grade_stem =  eleventh_grade_set[eleventh_grade_set['X2STU30OCC_STEM1'].isin([1,2,3])]
        eleventh_grade_non_stem =  eleventh_grade_set[~eleventh_grade_set['X2STU30OCC_STEM1'].isin([1,2,3])]
        return eleventh_grade_set, eleventh_grade_stem, eleventh_grade_non_stem

    def _find_persisters(self):
        """
        The paper notes that there are two groups, persisters and emergers.

        This utility class performs potential persisters filtering on dataframe.

        Note: persistors set produces the set of individuals who aspired to a STEM career
        in 9th grade. Not all continued to have that aspiration, and thus not all
        are "persisters" as the paper defines it.
        """
        persisters_set = self.dataframe[self.dataframe['X1STU30OCC_STEM1'].isin([1,2,3])]
        persisters = persisters_set[persisters_set['X2STU30OCC_STEM1'].isin([1,2,3])]
        non_persisters = persisters_set[~persisters_set['X2STU30OCC_STEM1'].isin([1,2,3])]
        return persisters_set, persisters, non_persisters
        
    
    def _find_emergers(self):
        """
        The paper notes that there are two groups, persisters and emergers.

        This utility class performs potential emergers filtering on dataframe.

        Note: this produces the set of individuals who did not aspire to a STEM career
        in 9th grade. Not all then showed that aspiration, and thus not all
        are "emergers" as the paper defines it.
        """
        emergers_set = self.dataframe[~self.dataframe['X1STU30OCC_STEM1'].isin([1,2,3])]
        emergers = emergers_set[emergers_set['X2STU30OCC_STEM1'].isin([1,2,3])]
        non_emergers = emergers_set[~emergers_set['X2STU30OCC_STEM1'].isin([1,2,3])]
        return emergers_set, emergers, non_emergers
    
    def table_b2(self):
        """
        Replicating summary results presented in b1.
        """
        ninth_grade_set, _, _ = self._find_ninth_grade()
        eleventh_grade_set, _, _ = self._find_eleventh_grade()
        persisters_set, _, _ = self._find_persisters()
        emergers_set, _, _ = self._find_emergers()

        x_axis = [('ninth_grade',ninth_grade_set, 'ninth_grade_aspirations'), 
                  ('eleventh_grade',eleventh_grade_set, 'eleventh_grade_aspirations'), 
                  ('persisters_grade',persisters_set, 'eleventh_grade_aspirations'), 
                  ('emergers_grade', emergers_set, 'eleventh_grade_aspirations')]
        
        results = {}
        for (name, x, bin_var) in x_axis:
            copy_x = x.copy()
            copy_x = copy_x.replace({'race': self.RACE_MAP,
                            'sex': self.SEX_MAP,
                            'HigherSES' : self.HSES_MAP,
                            bin_var: self.ASPIRATION_MAP})
            
            results[name +'_intersectional_results'] = copy_x.groupby(['race','HigherSES','sex', bin_var])

        full_intersectional_table = pd.DataFrame()
        for (name, x, bin_var) in x_axis:
            full_intersectional_table[name] = results[name +'_intersectional_results'].count()['SES']
        
        inter = full_intersectional_table.copy()
        for ind, row in inter.groupby(level=[0,1,2]).sum().iterrows():
            s = row
            s.name = ind
            s.name = s.name + tuple('n')
            full_intersectional_table = full_intersectional_table.append(s)

        full_intersectional_table = full_intersectional_table.reindex(self.TABLE_B2_REINDEX)
        results['full_intersectional_table'] = full_intersectional_table 
        
        # Create the high level aggregated tables from full_intersectional_table
        for i,name in enumerate(['race','HigherSES','sex']):
            n = full_intersectional_table.xs('n', level=3, axis=0, drop_level=False).groupby(level=i).sum()
            og_cols = n.columns.copy()
            n.columns = [col + '_n' for col in n.columns]
            yes = full_intersectional_table.xs('Yes', level=3, axis=0, drop_level=False).groupby(level=i).sum()
            yes.columns = [col + '_yes' for col in yes.columns]
            results[name + '_table'] = pd.concat([n, yes], axis=1)
            reorder_cols =[[col + '_n', col + '_yes'] for col in og_cols]
            reorder_cols = list(chain.from_iterable(reorder_cols))
            results[name + '_table'] = results[name + '_table'][reorder_cols]

        return results

    def figure_2(self):
        results = self.table_b2()
        table_b2 = results['full_intersectional_table']

        figure_2 = table_b2 / table_b2.groupby(level=[0,1,2]).sum()
        figure_2 = figure_2.xs('Yes', level=3, axis=0, drop_level=False)
        reindex = self.FIGURE_2_REINDEX.copy()
        reindex.reverse()
        figure_2 = figure_2.reindex(reindex)
        return figure_2.loc[reindex] # .plot(kind='barh', stacked=True, xlim=(0,0.5))