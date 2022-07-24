from meta_classes import Publication, Finding

import pandas as pd
import numpy as np

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

    FILENAME = 'saw2018cross'

    def __init__(self, dataframe=None, filename=None):
        if filename is not None:
            pd.read_pickle(filename)
        elif dataframe is not None:
            self.dataframe = dataframe
        else:
            self.dataframe = self._recreate_dataframe()

        self.FINDINGS.append(
            Finding(self.table_b2, description="Replicating TableB2.")
        )

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

            results[name + '_sex_n'] = x['sex'].value_counts()
            results[name + '_sex_yes'] = x.groupby(["sex", bin_var]).size()

            results[name + '_race_n'] = x['race'].value_counts()
            results[name + '_race_yes'] = x.groupby(["race", bin_var]).size()

            results[name + '_SES_n'] = x['SES'].value_counts()
            results[name + '_SES_yes'] = x.groupby(["SES", bin_var]).size()

            results[name +'_intersectional_results'] = x.group_by(['race','sex','HigherSES', bin_var])

        return results


