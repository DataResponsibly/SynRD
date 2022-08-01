from meta_classes import Publication, Finding, VisualFinding, FigureFinding

import pandas as pd
import numpy as np

from itertools import chain

class Saw2018Cross(Publication):
    """
    A class wrapper for all publication classes, for shared functionality.
    """
    DEFAULT_PAPER_ATTRIBUTES = {
        'id': 'saw2018cross',
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
            VisualFinding(self.table_b2, description="table_b2"),
            FigureFinding(self.figure_2, description="figure_2"),
            Finding(self.finding_526_1, description="finding_526_1",
                    text="""Among first-time 9th graders in fall 2009, only about 11.4% of 
                            students were interested in pursuing a STEM career upon entering 
                            high school (see Figure 1). The percentage declined slightly to 
                            10.0% for the same cohort of students after they spent their first 
                            three years in high school."""),
            Finding(self.finding_526_2, description="finding_526_2",
                    text="""First, the rates of interest in STEM professions dropped slightly 
                            for all racial/ethnic groups (ranging from 0.9% for Hispanics to 3.7% 
                            for multiracial students), except for Blacks."""),   
            Finding(self.finding_526_3, description="finding_526_3",
                    text="""Second, Black and Hispanic students consistently had significantly 
                            lower rates of interest (both in early 9th grade and late 11th grade) 
                            and persistence in STEM professions. At the end of 11th grade, for example, 
                            while 10.8% of Whites, 9.5% of Asians, and 11.6% of multiracial students 
                            aspired to a career in STEM, only 6.8% of Blacks and 8.2% of Hispanics did."""),              
        ]
        

        self.table_b2_dataframe = None
        
        

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

        self.table_b2_dataframe = results

        return results

    def table_b2_check(self):
        if self.table_b2_dataframe is None:
            results = self.table_b2()
            self.table_b2_dataframe = results
        else: 
            results = self.table_b2_dataframe
        return results

    def figure_2(self):
        results = self.table_b2_check()

        table_b2 = results['full_intersectional_table']

        figure_2 = table_b2 / table_b2.groupby(level=[0,1,2]).sum()
        figure_2 = figure_2.xs('Yes', level=3, axis=0, drop_level=False)
        reindex = self.FIGURE_2_REINDEX.copy()
        reindex.reverse()
        figure_2 = figure_2.reindex(reindex)
        return figure_2.loc[reindex] # .plot(kind='barh', stacked=True, xlim=(0,0.5))

    def finding_526_1(self):
        """
        Among first-time 9th graders in fall 2009, only about 11.4% of 
        students were interested in pursuing a STEM career upon entering 
        high school (see Figure 1). The percentage declined slightly to 
        10.0% for the same cohort of students after they spent their first 
        three years in high school.
        """
        results = self.table_b2_check()

        ng_yes = sum(results['sex_table']['ninth_grade_yes']) 
        ng_total = sum(results['sex_table']['ninth_grade_n'])
        interest_stem_ninth = ng_yes / ng_total
        eg_yes = sum(results['sex_table']['eleventh_grade_yes']) 
        eg_total = sum(results['sex_table']['eleventh_grade_n'])
        interest_stem_eleventh = eg_yes / eg_total
        # Check soft assertion from paper
        soft_finding = interest_stem_ninth > interest_stem_eleventh
        # Check relative difference magnitude
        hard_finding = interest_stem_ninth - interest_stem_eleventh
        return ([interest_stem_ninth,interest_stem_eleventh], soft_finding, [hard_finding])

    def finding_526_2(self):
        """
        First, the rates of interest in STEM professions dropped slightly 
        for all racial/ethnic groups (ranging from 0.9% for Hispanics to 3.7% 
        for multiracial students), except for Blacks.
        """
        results = self.table_b2_check()

        soft_finding = True
        hard_findings = []
        all_findings = []
        for race in list(results['race_table'].index):
            race_vals = results['race_table'].loc[race]
            percent_ninth = race_vals['ninth_grade_yes'] / race_vals['ninth_grade_n']
            percent_eleventh = race_vals['eleventh_grade_yes'] / race_vals['eleventh_grade_n']
            if percent_ninth < percent_eleventh:
                soft_finding = False
            hard_findings.append(percent_eleventh - percent_ninth)
            all_findings.append([percent_ninth,percent_eleventh])
        
        return (all_findings, soft_finding, hard_findings)

    def finding_526_3(self):
        """
        Second, Black and Hispanic students consistently had significantly 
        lower rates of interest (both in early 9th grade and late 11th grade) 
        and persistence in STEM professions. At the end of 11th grade, for example, 
        while 10.8% of Whites, 9.5% of Asians, and 11.6% of multiracial students 
        aspired to a career in STEM, only 6.8% of Blacks and 8.2% of Hispanics did.
        """
        results = self.table_b2_check()

        soft_finding = True
        # Higher interest set
        white_interest = results['race_table'].loc['White']
        white_interest = white_interest['eleventh_grade_yes'] / white_interest['eleventh_grade_n']
        asian_interest = results['race_table'].loc['Asian']
        asian_interest = asian_interest['eleventh_grade_yes'] / asian_interest['eleventh_grade_n']
        multiracial_interest = results['race_table'].loc['Multiracial']
        multiracial_interest = multiracial_interest['eleventh_grade_yes'] / multiracial_interest['eleventh_grade_n']
        set_1 = [white_interest,asian_interest,multiracial_interest]
        # Lower interest set
        black_interest = results['race_table'].loc['Black']
        black_interest = black_interest['eleventh_grade_yes'] / black_interest['eleventh_grade_n']
        hispanic_interest = results['race_table'].loc['Hispanic']
        hispanic_interest = hispanic_interest['eleventh_grade_yes'] / hispanic_interest['eleventh_grade_n']
        set_2 = [black_interest,hispanic_interest]
        # To softly validate the claim, min(set_1) should be greater than max(set_2)
        soft_finding = min(set_1) > max(set_2)
        # For hard finding validation, we need the difference to be at least as much as
        # reported in the paper
        hard_finding_value = abs(min(set_1) - max(set_2))
        hard_finding_bool = hard_finding_value >= 0.013
        return ([set_1, set_2], soft_finding, [hard_finding_value, hard_finding_bool])