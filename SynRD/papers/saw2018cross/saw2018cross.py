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
        'current_citations': 67,
        'base_dataframe_pickle': 'saw2018cross_dataframe.pickle'
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

    SES_MAP = {
        1: "low-SES",
        2: "low-middle-SES",
        3: "high-middle-SES",
        4: "high-SES",
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
            Finding(self.finding_526_4, description="finding_526_4",
                    text="""Third, interestingly, among those who previously were not interested 
                            in STEM fields, students from all racial/ethnic backgrounds, except 
                            Blacks (5.1%), gained interest in STEM jobs at a similar rate 
                            (about 7%) after spending three years in high school."""),
            Finding(self.finding_526_5, description="finding_526_5",
                    text="""Of those who aspired to a career in STEM in early 9th grade, 
                            only about 34.3% maintained their interest until late 11th 
                            grade (or persisters). Interestingly, about 6.9% students 
                            (or emergers) who initially did not aspire to STEM careers 
                            developed an interest three years after enrolling in high 
                            school."""),    
            Finding(self.finding_526_6, description="finding_526_6",
                    text="""The patterns of SES disparities were clear and quite 
                            consistent across multiple indicators of cross-sectional 
                            and longitudinal STEM career aspirations. Students with 
                            lower SES were less likely to aspire to a STEM career at 
                            the start and toward the end of high school. In late 11th 
                            grade, for instance, while about 14.4% of high SES students 
                            aspired to pursue a career in STEM, only 10.6% of 
                            high-middle SES, 9.0% of low-middle SES, and 7.1% of low SES 
                            students did."""),     
            Finding(self.finding_526_7, description="finding_526_7",
                    text="""Although the persistence and emergence rates are fairly 
                            low, the absolute numbers of nonpersisters (unweighted 
                            1,272 out of 1,988) and emergers (unweighted 1,132 out of 
                            14,941) are more or less identical, which explains the 
                            quite stable rates of STEM career aspirations among high 
                            school students over time."""), 
            Finding(self.finding_526_8, description="finding_526_8",
                    text="""As shown in Figure 1 (for regression estimates, see 
                            Appendix Table B1), considerable cross-sectional and 
                            longitudinal disparities in STEM career aspirations 
                            existed among gender, racial/ethnic, and SES groups. 
                            Gender and SES gaps in STEM career aspirations appear 
                            to be widening over time, whereas the racial/ethnic gaps 
                            seem to be closing. At the beginning of 9th grade, 
                            about 14.5% of boys and 8.4% of girls were interested 
                            in a STEM career (a 6.1% gap). At the end of 11th grade, 
                            the corresponding percentages were 14.7% and 5.3%, 
                            suggesting that the gender gap grew to 9.4 percentage points."""),
            Finding(self.finding_526_9, description="finding_526_9",
                    text="""From a longitudinal perspective, students from the two 
                            lower SES groups—low-middle and low SES groups—had 
                            significantly fewer persisters (31.9% and 29.9%) and 
                            emergers (6.1% and 5.4%) than their high SES peers 
                            (45.1% and 9.0%, respectively)."""),  
            Finding(self.finding_526_10, description="finding_526_10",
                    text="""The growing gender gap resulted from the lower percentage 
                            of persisters (24.6%) as well as the lower percentage of 
                            emergers (3.7%) among girls throughout the first three 
                            years of high school. For boys, the corresponding percentages 
                            were 40.0% and 10.5%."""),  
            Finding(self.finding_527_1, description="finding_527_1",
                    text="""For Whites, the patterns of cross-sectional and longitudinal 
                            disparities in STEM career aspirations across gender and SES 
                            groups were prominent and consistent. In particular, higher 
                            SES boys reported the highest rates of all four indicators 
                            of STEM career aspirations, followed by lower SES boys, 
                            higher SES girls, and lower SES girls."""),  
            Finding(self.finding_527_4, description="finding_527_4",
                    text="""First, compared with their White counterparts (interracial 
                            but intragender and intra-SES comparisons), higher SES boys 
                            from Black, Asian, and multiracial groups showed similar 
                            levels of STEM career aspirations in nearly all indicators, 
                            except that higher SES boys from the Hispanic group reported 
                            lower levels of career aspirations in STEM in those indicators."""),  
            Finding(self.finding_527_5, description="finding_527_5",
                    text="""For example, while 17.9% of White boys from higher SES 
                            families aspired to a career in STEM upon entering high 
                            school, only 1.8% of Black girls from lower SES families 
                            did (a 16.1% gap)."""),  
            Finding(self.finding_527_6, description="finding_527_6",
                    text="""From high school freshman to junior year, the gaps in 
                            STEM career aspirations between White boys from higher 
                            SES households and girls from all racial/ethnic groups, 
                            regardless of their SES, on average grew by 6.6 percentage 
                            points."""),
            Finding(self.finding_527_7, description="finding_527_7",
                    text="""Second, compared with their White high SES peers 
                            (intragender but interracial and inter-SES comparisons), 
                            Asian boys, though raised in lower SES households, had 
                            comparable rates of STEM career aspirations, unlike Black, 
                            Hispanic, and multiracial boys from lower SES families who 
                            consistently had significantly lower rates of all four 
                            indicators."""),
            Finding(self.finding_527_8, description="finding_527_8",
                    text="""In terms of persisters, whereas nearly half of White boys 
                            from higher SES families (46.6%) who initially had a career 
                            interest in STEM maintained their interest, only about 14.0% 
                            of Black boys from lower SES group, Hispanic girls from 
                            higher SES group, and Asian girls from lower SES group did."""),
            Finding(self.finding_527_9, description="finding_527_9",
                    text="""Third, compared with White higher SES boys, girls from Black, 
                            Hispanic, Asian, and multiracial groups, regardless of their 
                            SES, had significantly lower rates of almost all four 
                            indicators of STEM career aspirations in high school.""")
                                   
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

    def _granular_SES_dataframe(self):
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
            copy_x = copy_x.replace({'SES': self.SES_MAP})
            
            results[name +'_ses_results'] = copy_x.groupby(['SES', bin_var])

        ses_df = pd.DataFrame()
        for (name, x, bin_var) in x_axis:
            ses_df[name] = results[name +'_ses_results'].count()['sex']
        return ses_df

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
    
    def finding_526_4(self):
        """
        Third, interestingly, among those who previously were not interested 
        in STEM fields, students from all racial/ethnic backgrounds, except 
        Blacks (5.1%), gained interest in STEM jobs at a similar rate 
        (about 7%) after spending three years in high school.

        NOTE: Finding not directly reproducible from paper with public
        data
        """
        results = self.table_b2_check()

        # Similar interest set (seemingly defined as within approx 1% of each other)
        white_interest = results['race_table'].loc['White']
        white_interest = white_interest['emergers_grade_yes'] / white_interest['emergers_grade_n']
        asian_interest = results['race_table'].loc['Asian']
        asian_interest = asian_interest['emergers_grade_yes'] / asian_interest['emergers_grade_n']
        multiracial_interest = results['race_table'].loc['Multiracial']
        multiracial_interest = multiracial_interest['emergers_grade_yes'] / multiracial_interest['emergers_grade_n']
        similar_set = [white_interest, 
                        asian_interest,
                        multiracial_interest]
        # Dissimilar interest set (seemingly defined as within approx 1% of each other and 1.5% from other group mean)
        black_interest = results['race_table'].loc['Black']
        black_interest = black_interest['emergers_grade_yes'] / black_interest['emergers_grade_n']
        hispanic_interest = results['race_table'].loc['Hispanic']
        hispanic_interest = hispanic_interest['emergers_grade_yes'] / hispanic_interest['emergers_grade_n']
        dissimilar_set = [black_interest,
                        hispanic_interest]
        # Essentially, this finding is saying that levels of emergers were
        # "similar" for all but black (and hispanic) 
        # individuals in the sample, which was lower by more than 1.5%
        similar_set_mean = np.mean(similar_set)
        dissimilar_set_mean = np.mean(dissimilar_set)
        hard_findings_values = [interest - similar_set_mean for interest in similar_set] + \
                               [interest - dissimilar_set_mean for interest in dissimilar_set]
        soft_finding_bool = (similar_set_mean - dissimilar_set_mean) >= 0.015
        return ([similar_set, dissimilar_set], soft_finding_bool, hard_findings_values)
    
    def finding_526_5(self):
        """
        Of those who aspired to a career in STEM in early 9th grade, 
        only about 34.3% maintained their interest until late 11th 
        grade (or persisters). Interestingly, about 6.9% students 
        (or emergers) who initially did not aspire to STEM careers 
        developed an interest three years after enrolling in high 
        school.
        """
        results = self.table_b2_check()
        
        # Here we are saying that roughly 34% of individuals with interest
        # in 9th grade "persisted" with interest
        p = sum(results['sex_table']['persisters_grade_yes']) /\
            sum(results['sex_table']['persisters_grade_n']) 
        
        # While only roughly 6% of individuals without interest
        # in 9th grade "emerged"with interest
        e = sum(results['sex_table']['emergers_grade_yes']) /\
            sum(results['sex_table']['emergers_grade_n'])
        
        # Here soft check is are each of these values within ~2% of reported
        # values
        soft_check = (abs(p - 0.34) < 0.02) and (abs(e - 0.06) < 0.02)

        return ([], soft_check, [p,e])

    def finding_526_6(self):
        """
        The patterns of SES disparities were clear and quite 
        consistent across multiple indicators of cross-sectional 
        and longitudinal STEM career aspirations. Students with 
        lower SES were less likely to aspire to a STEM career at 
        the start and toward the end of high school. In late 11th 
        grade, for instance, while about 14.4% of high SES students 
        aspired to pursue a career in STEM, only 10.6% of 
        high-middle SES, 9.0% of low-middle SES, and 7.1% of low SES 
        students did.
        """
        ses_df = self._granular_SES_dataframe()
        
        df_ses_percentages = pd.DataFrame()
        for ses_type in list(self.SES_MAP.values()):
            row = ses_df.xs((ses_type, 1), level=[0,1], axis=0, drop_level=False).groupby(level=0).sum()/\
                ses_df.xs((ses_type, 0), level=[0,1], axis=0, drop_level=False).groupby(level=0).sum()
            df_ses_percentages = df_ses_percentages.append(row)
        
        checks_per_column = []
        percentages = []
        for col in df_ses_percentages.columns:
            p = list(df_ses_percentages[col])
            # Allow there to be a single flipped expected ordering
            checks_per_column.append([(p[k+1] - p[k])>0\
                                       for k in range(len(p) - 1)].count(True)\
                                       == len(p)-2)
            percentages = percentages + p

        soft_finding = all(checks_per_column)
        return ([], soft_finding, percentages)

    def finding_526_7(self):
        """
        Although the persistence and emergence rates are fairly 
        low, the absolute numbers of nonpersisters (unweighted 
        1,272 out of 1,988) and emergers (unweighted 1,132 out of 
        14,941) are more or less identical, which explains the 
        quite stable rates of STEM career aspirations among high 
        school students over time.
        """
        results = self.table_b2_check()

        absolute_nonpersisters = sum(results['sex_table']['persisters_grade_n'])\
                                 - sum(results['sex_table']['persisters_grade_yes'])
        
        emergers = sum(results['sex_table']['emergers_grade_yes']) 

        # Difference in reported absolute numbers is 140 (approx 150)
        # in reported so here we do 150 
        soft_finding = abs(absolute_nonpersisters - emergers) <= 150
        return ([], soft_finding, [absolute_nonpersisters, emergers])

    def finding_526_8(self):
        """
        As shown in Figure 1 (for regression estimates, see 
        Appendix Table B1), considerable cross-sectional and 
        longitudinal disparities in STEM career aspirations 
        existed among gender, racial/ethnic, and SES groups. 
        Gender and SES gaps in STEM career aspirations appear 
        to be widening over time, whereas the racial/ethnic gaps 
        seem to be closing. At the beginning of 9th grade, 
        about 14.5% of boys and 8.4% of girls were interested 
        in a STEM career (a 6.1% gap). At the end of 11th grade, 
        the corresponding percentages were 14.7% and 5.3%, 
        suggesting that the gender gap grew to 9.4 percentage points.
        """
        results = self.table_b2_check()

        p_9 = results['sex_table']['ninth_grade_yes'] /\
              results['sex_table']['ninth_grade_n']
        p_11 = results['sex_table']['eleventh_grade_yes'] /\
               results['sex_table']['eleventh_grade_n']
        difs = p_11 - p_9
        # Did the gap grow?
        soft_finding = difs['boys'] > difs['girls']
        return ([], soft_finding, [difs['boys'], difs['girls']])

    def finding_526_9(self):
        """
        From a longitudinal perspective, students from the two 
        lower SES groups—low-middle and low SES groups—had 
        significantly fewer persisters (31.9% and 29.9%) and 
        emergers (6.1% and 5.4%) than their high SES peers 
        (45.1% and 9.0%, respectively).
        """
        ses_df = self._granular_SES_dataframe()

        df_ses_percentages = pd.DataFrame()
        for ses_type in list(self.SES_MAP.values()):
            row = ses_df.xs((ses_type, 1), level=[0,1], axis=0, drop_level=False).groupby(level=0).sum()/\
                ses_df.xs((ses_type, 0), level=[0,1], axis=0, drop_level=False).groupby(level=0).sum()
            df_ses_percentages = df_ses_percentages.append(row)
        
        # Significant here is 5% and 2% (proportional to scale)
        diff = 0.05
        low_ses_p = df_ses_percentages['persisters_grade'].loc['low-SES']
        low_middle_ses_p = df_ses_percentages['persisters_grade'].loc['low-middle-SES']
        high_ses_p = df_ses_percentages['persisters_grade'].loc['high-SES']
        soft_check_1 = (high_ses_p > low_middle_ses_p + diff) and (high_ses_p > low_ses_p + diff)

        diff = 0.02
        low_ses_e = df_ses_percentages['emergers_grade'].loc['low-SES']
        low_middle_ses_e = df_ses_percentages['emergers_grade'].loc['low-middle-SES']
        high_ses_e = df_ses_percentages['emergers_grade'].loc['high-SES']
        soft_check_2 = (high_ses_e > low_middle_ses_e + diff) and (high_ses_e > low_ses_e + diff)

        soft_check = soft_check_1 and soft_check_2
        return ([], soft_check, [low_ses_p,low_middle_ses_p,high_ses_p,low_ses_e,low_middle_ses_e,high_ses_e])


    def finding_526_10(self):
        """
        The growing gender gap resulted from the lower percentage 
        of persisters (24.6%) as well as the lower percentage of 
        emergers (3.7%) among girls throughout the first three 
        years of high school. For boys, the corresponding percentages 
        were 40.0% and 10.5%.
        """
        results = self.table_b2_check()

        # Boys
        p_b = results['sex_table']['persisters_grade_yes'].loc['boys'] /\
            results['sex_table']['persisters_grade_n'].loc['boys'] 

        e_b = results['sex_table']['emergers_grade_yes'].loc['boys'] /\
            results['sex_table']['emergers_grade_n'].loc['boys']

        # Girls 
        p_g = results['sex_table']['persisters_grade_yes'].loc['girls'] /\
            results['sex_table']['persisters_grade_n'].loc['girls'] 

        e_g = results['sex_table']['emergers_grade_yes'].loc['girls'] /\
            results['sex_table']['emergers_grade_n'].loc['girls']
        
        soft_check = (p_b > p_g) and (e_b > e_g)

        return ([], soft_check, [p_b, e_b, p_g, e_g])
        

    def finding_527_1(self):
        """
        For Whites, the patterns of cross-sectional and longitudinal 
        disparities in STEM career aspirations across gender and SES 
        groups were prominent and consistent. In particular, higher 
        SES boys reported the highest rates of all four indicators 
        of STEM career aspirations, followed by lower SES boys, 
        higher SES girls, and lower SES girls.
        """
        results = self.table_b2_check()

        w_hses_b = results['full_intersectional_table'].loc[('White','higher-SES','boys','Yes')] /\
                    results['full_intersectional_table'].loc[('White','higher-SES','boys','n')]
        w_lses_b = results['full_intersectional_table'].loc[('White','lower-SES','boys','Yes')] /\
                    results['full_intersectional_table'].loc[('White','lower-SES','boys','n')]
        w_hses_g = results['full_intersectional_table'].loc[('White','higher-SES','girls','Yes')] /\
                    results['full_intersectional_table'].loc[('White','higher-SES','girls','n')]
        w_lses_g = results['full_intersectional_table'].loc[('White','lower-SES','girls','Yes')] /\
                    results['full_intersectional_table'].loc[('White','lower-SES','girls','n')]
        
        # Here we want the pattern to be true for 10/12 of the results for "consistent"
        soft_check = sum(list(w_hses_b > w_lses_b) + list(w_lses_b > w_hses_g) + list(w_hses_g > w_lses_g)) >= 10

        return ([], soft_check, [w_hses_b.array, w_lses_b.array, w_hses_g.array, w_lses_g.array])

    def finding_527_2(self):
        """
        No clear-cut patterns emerged when analyzing the differences 
        in STEM career aspirations across gender and SES groups for 
        Black, Hispanic, Asian, and multiracial students.
        """
        # TODO: determine if this "finding" is replicable - conclusion?
        pass

    def finding_527_3(self):
        """
        The gaps in STEM career aspirations between White, higher 
        SES boys (reference group for this second set of analyses) 
        and some of their counterparts from other intersectional 
        groups were strikingly large and widening over time (see 
        Figure 2; Appendix Table B3 reports regression estimates).
        """
        # TODO: determine if this "finding" is replicable - conclusion?
        pass

    def finding_527_4(self):
        """
        First, compared with their White counterparts (interracial 
        but intragender and intra-SES comparisons), higher SES boys 
        from Black, Asian, and multiracial groups showed similar 
        levels of STEM career aspirations in nearly all indicators, 
        except that higher SES boys from the Hispanic group reported 
        lower levels of career aspirations in STEM in those indicators.
        """
        # Indicators here are 9th/11th grade stem aspirations
        # NOTE: the finding for hispanic groups was not clear, and
        # so is not accounted for here. This finding is simply that
        # higher SES boys have similar STEM aspirations (within ~5-6%)
        results = self.table_b2_check()
        
        grades = ['ninth_grade','eleventh_grade']
        diff = 0.06
        w_hses_b = results['full_intersectional_table'][grades].loc[('White','higher-SES','boys','Yes')] /\
                    results['full_intersectional_table'][grades].loc[('White','higher-SES','boys','n')]

        b_hses_b = results['full_intersectional_table'][grades].loc[('Black','higher-SES','boys','Yes')] /\
                    results['full_intersectional_table'][grades].loc[('Black','higher-SES','boys','n')]

        h_hses_b = results['full_intersectional_table'][grades].loc[('Hispanic','higher-SES','boys','Yes')] /\
                    results['full_intersectional_table'][grades].loc[('Hispanic','higher-SES','boys','n')]

        a_hses_b = results['full_intersectional_table'][grades].loc[('Asian','higher-SES','boys','Yes')] /\
                    results['full_intersectional_table'][grades].loc[('Asian','higher-SES','boys','n')]

        m_hses_b = results['full_intersectional_table'][grades].loc[('Multiracial','higher-SES','boys','Yes')] /\
                    results['full_intersectional_table'][grades].loc[('Multiracial','higher-SES','boys','n')]
        
        soft_check = sum(list(abs(w_hses_b - b_hses_b) < diff) +
                     list(abs(w_hses_b - h_hses_b) < diff) + 
                     list(abs(w_hses_b - a_hses_b) < diff) +
                     list(abs(w_hses_b - m_hses_b) < diff)) == 8
        
        return ([], soft_check, [w_hses_b.array, 
                                b_hses_b.array, 
                                h_hses_b.array, 
                                a_hses_b.array,
                                m_hses_b.array])

    def finding_527_5(self):
        """
        For example, while 17.9% of White boys from higher SES 
        families aspired to a career in STEM upon entering high 
        school, only 1.8% of Black girls from lower SES families 
        did (a 16.1% gap).
        """
        results = self.table_b2_check()

        w_hses_b = results['full_intersectional_table'].loc[('White','higher-SES','boys','Yes')]['ninth_grade'] /\
                    results['full_intersectional_table'].loc[('White','higher-SES','boys','n')]['ninth_grade']

        b_lses_g = results['full_intersectional_table'].loc[('Black','lower-SES','girls','Yes')]['ninth_grade'] /\
                    results['full_intersectional_table'].loc[('Black','lower-SES','girls','n')]['ninth_grade']

        gap = w_hses_b - b_lses_g
        # Significant difference here is considered to be at least
        # 10% (replicated difference was 11.6)
        soft_finding = gap > 0.10

        return ([], soft_finding, [w_hses_b, b_lses_g, gap])

    def finding_527_6(self):
        """
        From high school freshman to junior year, the gaps in 
        STEM career aspirations between White boys from higher 
        SES households and girls from all racial/ethnic groups, 
        regardless of their SES, on average grew by 6.6 percentage 
        points.
        """
        results = self.table_b2_check()

        w_hses_b_9 = results['full_intersectional_table'].loc[('White','higher-SES','boys','Yes')]['ninth_grade'] /\
            results['full_intersectional_table'].loc[('White','higher-SES','boys','n')]['ninth_grade']
        w_hses_b_11 = results['full_intersectional_table'].loc[('White','higher-SES','boys','Yes')]['eleventh_grade'] /\
                    results['full_intersectional_table'].loc[('White','higher-SES','boys','n')]['eleventh_grade']

        gap_growths = []
        for r in self.RACE_MAP.values():
            for ses in self.HSES_MAP.values():
                g_9 = results['full_intersectional_table'].loc[(r,ses,'girls','Yes')]['ninth_grade'] /\
                    results['full_intersectional_table'].loc[(r,ses,'girls','n')]['ninth_grade']
                gap_9 = w_hses_b_9 - g_9
                
                g_11 = results['full_intersectional_table'].loc[(r,ses,'girls','Yes')]['eleventh_grade'] /\
                    results['full_intersectional_table'].loc[(r,ses,'girls','n')]['eleventh_grade']
                gap_11 = w_hses_b_11 - g_11
                growth = gap_11 - gap_9
                gap_growths.append(growth)

        # The soft finding here is just that, on average the gaps grew
        # by a reasonable margin
        soft_finding = np.mean(gap_growths) > 0.01
        return ([], soft_finding, gap_growths)

    def finding_527_7(self):
        """
        Second, compared with their White high SES peers 
        (intragender but interracial and inter-SES comparisons), 
        Asian boys, though raised in lower SES households, had 
        comparable rates of STEM career aspirations, unlike Black, 
        Hispanic, and multiracial boys from lower SES families who 
        consistently had significantly lower rates of all four 
        indicators.
        """
        results = self.table_b2_check()

        w_hses_b = results['full_intersectional_table'].loc[('White','higher-SES','boys','Yes')] /\
            results['full_intersectional_table'].loc[('White','higher-SES','boys','n')]

        a_lses_b = results['full_intersectional_table'].loc[('Asian','lower-SES','boys','Yes')] /\
                    results['full_intersectional_table'].loc[('Asian','lower-SES','boys','n')]

        b_lses_b = results['full_intersectional_table'].loc[('Black','lower-SES','boys','Yes')] /\
                    results['full_intersectional_table'].loc[('Black','lower-SES','boys','n')]

        h_lses_b = results['full_intersectional_table'].loc[('Hispanic','lower-SES','boys','Yes')] /\
                    results['full_intersectional_table'].loc[('Hispanic','lower-SES','boys','n')]

        m_lses_b = results['full_intersectional_table'].loc[('Multiracial','lower-SES','boys','Yes')] /\
                    results['full_intersectional_table'].loc[('Multiracial','lower-SES','boys','n')]

        check_white_asian_similar = sum(list(abs(w_hses_b - a_lses_b) < 0.046)) >= 3

        check_other_disimilar = sum(list(abs(w_hses_b - b_lses_b) > 0.046) +\
            list(abs(w_hses_b - h_lses_b) > 0.046) + \
            list(abs(w_hses_b - m_lses_b) > 0.046)) >= 10
        
        soft_finding = check_white_asian_similar and check_other_disimilar

        return ([], soft_finding, [w_hses_b.array, 
                                a_lses_b.array, 
                                b_lses_b.array, 
                                h_lses_b.array,
                                m_lses_b.array])


    def finding_527_8(self):
        """
        In terms of persisters, whereas nearly half of White boys 
        from higher SES families (46.6%) who initially had a career 
        interest in STEM maintained their interest, only about 14.0% 
        of Black boys from lower SES group, Hispanic girls from 
        higher SES group, and Asian girls from lower SES group did.
        """
        results = self.table_b2_check()

        w_hses_b = results['full_intersectional_table'].loc[('White','higher-SES','boys','Yes')]['persisters_grade'] /\
                    results['full_intersectional_table'].loc[('White','higher-SES','boys','n')]['persisters_grade']

        b_lses_b = results['full_intersectional_table'].loc[('Black','lower-SES','boys','Yes')]['persisters_grade'] /\
                    results['full_intersectional_table'].loc[('Black','lower-SES','boys','n')]['persisters_grade']

        h_lses_g = results['full_intersectional_table'].loc[('Hispanic','higher-SES','girls','Yes')]['persisters_grade'] /\
                    results['full_intersectional_table'].loc[('Hispanic','higher-SES','girls','n')]['persisters_grade']

        a_lses_g = results['full_intersectional_table'].loc[('Asian','lower-SES','girls','Yes')]['persisters_grade'] /\
                    results['full_intersectional_table'].loc[('Asian','lower-SES','girls','n')]['persisters_grade']

        near_half_check = w_hses_b + 0.10 > 0.5

        less_than_a_quarter_check = (b_lses_b < 0.25) and (h_lses_g < 0.25) and (a_lses_g < 0.25)

        soft_finding = near_half_check and less_than_a_quarter_check

        return ([], soft_finding, [w_hses_b, b_lses_b, h_lses_g, a_lses_g])

    def finding_527_9(self):
        """
        Third, compared with White higher SES boys, girls from Black, 
        Hispanic, Asian, and multiracial groups, regardless of their 
        SES, had significantly lower rates of almost all four 
        indicators of STEM career aspirations in high school.
        """
        results = self.table_b2_check()

        w_hses_b = results['full_intersectional_table'].loc[('White','higher-SES','boys','Yes')] /\
            results['full_intersectional_table'].loc[('White','higher-SES','boys','n')]

        comparisons = []
        for r in ['Black','Hispanic','Asian','Multiracial']:
            for ses in ['lower-SES','higher-SES']:
                group = results['full_intersectional_table'].loc[(r,ses,'girls','Yes')] /\
                    results['full_intersectional_table'].loc[(r,ses,'girls','n')]
                comparisons.append(list(w_hses_b > group))
        comparisons = [i for l in comparisons for i in l]
        soft_finding = sum(comparisons) >= (len(comparisons) - 2)

        return ([], soft_finding, comparisons)