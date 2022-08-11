from meta_classes import Publication, Finding, VisualFinding

import pandas as pd
import numpy as np

from statsmodels.regression.linear_model import WLS

# Note: properly installing lightgbm allows you to run miceforest. If you have an M1 mac, please see:
# https://towardsdatascience.com/install-xgboost-and-lightgbm-on-apple-m1-macs-cb75180a2dda
import miceforest as mf
from math import nan

class Lee2021Ability(Publication):
    """
    A class wrapper for all publication classes, for shared functionality.
    """
    DEFAULT_PAPER_ATTRIBUTES = {
        'id': 'lee2021ability',
        'length_pages': 10,
        'authors': ['Glona Lee', 'Sandra D. Simpkins'],
        'journal': 'Journal of Adolescence',
        'year': 2021,
        'current_citations': 2,
        'base_dataframe_pickle': 'lee2021ability_dataframe.pickle'
    }

    RACE_MAP = {
        0: "White",
        1: "Black",
        2: "Hispanic",
        3: "Asian",
        4: "Other",
        5: "Other"
    }

    SEX_MAP = {
        0: "Male",
        1: "Female"
    }

    FILENAME = 'lee2021ability'

    ALL_COLUMNS = [
        'S2MSPR12',
        'S1MFALL09',
        'S2MTCHTREAT', 
        'S2MTCHINTRST',
        'S2MTCHEASY',
        'S2MTCHTHINK',
        'S2MTCHGIVEUP',
        'S1MTESTS',
        'S1MTEXTBOOK',
        'S1MSKILLS',
        'S1MASSEXCL',
        'P1MUSEUM',
        'P2MUSEUM',
        'P1COMPUTER',
        'P2COMPUTER',
        'P1FIXED',
        'P2FIXED',
        'P1LIBRARY',
        'P2LIBRARY',
        'P1STEMDISC',
        'P2STEMDISC',
        'X2TXMSCR',
        'X2X1TXMSCR',
        'X1SEX',
        'X1RACE',
        'X1SES_U',
        'X3THIMATH9',
        'X1TXMSCR',
        'W1PARENT'
    ]

    WEIGHTS = None

    corr_df = None

    regression_df = None

    cov_df = None

    def __init__(self, dataframe=None, filename=None):
        if filename is not None:
            self.dataframe = pd.read_pickle(filename)
        elif dataframe is not None:
            self.dataframe = dataframe
        else:
            self.dataframe = self._recreate_dataframe()
        
        self.FINDINGS = self.FINDINGS + [
            VisualFinding(self.table_2, description="table_2"),
            VisualFinding(self.table_3, description="table_3"),
            Finding(self.finding_52_1, description="finding_52_1",
                    text="""As predicted, a negative correlation was found
                            between perceived low math teacher support and 11th 
                            grade math achievement (r = -0.11)."""), 
            Finding(self.finding_52_2, description="finding_52_2",
                    text="""Ability self-concepts and parental support 
                            in 9th grade were positively correlated with students' 
                            11th grade achievement (r = 0.30 and r = 0.12 respectively)."""), 
            Finding(self.finding_52_3, description="finding_52_3",
                    text="""A strong positive correlation was found between 
                            9th and 11th grade math achievement (r = 0.74)."""),
            Finding(self.finding_54_1, description="finding_54_1",
                    text="""perceived low math teacher support in 11th grade 
                            negatively predicted students' 11th grade math achievement 
                            (B = -1.51, p &lt; .001) while controlling for students' 
                            demographics, 9th grade math achievement score, and math 
                            course."""), 
            Finding(self.finding_54_2, description="finding_54_2",
                    text="""Second, as shown under Model 2, math ability self-concepts 
                            positively and directly predicted 11th grade math achievement 
                            (B = 2.51, p &lt; .001)."""), 
            Finding(self.finding_54_3, description="finding_54_3",
                    text="""Contrary to our hypothesis and as shown under Model 3, 
                            the relation between perceived low math teacher support and 
                            math achievement was not moderated by adolescents' math 
                            ability self-concepts (B = -0.46, p = .329)"""), 
            Finding(self.finding_54_4, description="finding_54_4",
                    text="""Third, as hypothesized and shown under Model 5, the 3-way 
                            interaction was found to be statistically significant 
                            (B = -4.38, p &lt; .05). That is, the interaction between 
                            perceived low math teacher support and adolescents' math 
                            ability self-concepts in predicting adolescents' math 
                            achievement varied by the level of parental support."""), 
            Finding(self.finding_54_5, description="finding_54_5",
                    text="""As expected, perceived low teacher support was linked 
                            to lower achievement when adolescents were low on both 
                            protective factors, namely low ability selfconcepts and 
                            low parental support (B= -2.23, p = .003)."""), 
        ]
    
    def _recreate_dataframe(self, filename='lee2021ability_dataframe.pickle'):
        student_survey = pd.read_csv('lee2021ability/data/36423-0002-Data.tsv',sep='\t')

        # Math enrollment
        student_survey = student_survey[(student_survey['S2MSPR12'] == 1) & (student_survey['S1MFALL09'] == 1)]

        low_teacher_support = [
            ('S2MTCHTREAT','treats some kids better'),
            ('S2MTCHINTRST','makes math interesting'),
            ('S2MTCHEASY','makes math easy to understand'),
            ('S2MTCHTHINK','wants students to think'),
            ('S2MTCHGIVEUP','doesnt let students give up')
        ]

        ability_self_concept = [
            ('S1MTESTS','confident can do excellent job on test'),
            ('S1MTEXTBOOK','certain can understand math textbook'),
            ('S1MSKILLS','certain can master math skills'),
            ('S1MASSEXCL','confident can do excellent job on assignments')
        ]

        parental_support = [
            ('P1MUSEUM','went to science or engineering museum'),
            ('P2MUSEUM','went to science or engineering museum'),
            ('P1COMPUTER','worked or played on computer'),
            ('P2COMPUTER','worked or played on computer'),
            ('P1FIXED','built or fixed something'),
            ('P2FIXED','built or fixed something'),
            ('P1LIBRARY','visited a library'),
            ('P2LIBRARY','visited a library'),
            ('P1STEMDISC','discussed STEM program or article'),
            ('P2STEMDISC','discussed STEM program or article'),]

        math_acheivement_score = [('X2TXMSCR', 'score')]

        highest_level_math = [('X3THIMATH9', 'level')]

        base_year_score = [('X1TXMSCR', 'base_score')]

        student_survey = student_survey[self.ALL_COLUMNS]

        student_survey['sex'] = student_survey['X1SEX']
        student_survey.loc[student_survey["sex"] == 1, "sex"] = 0
        student_survey.loc[student_survey["sex"] == 2, "sex"] = 1
        student_survey['sex'] = student_survey['sex'].astype(np.int8)

        student_survey['race'] = student_survey['X1RACE']
        student_survey.loc[student_survey["race"] == 8, "race"] = 10
        student_survey.loc[student_survey["race"] == 3, "race"] = 11
        student_survey.loc[student_survey["race"] == 4, "race"] = 12
        student_survey.loc[student_survey["race"] == 5, "race"] = 12
        student_survey.loc[student_survey["race"] == 2, "race"] = 13
        student_survey.loc[student_survey["race"] == 6, "race"] = 14
        student_survey.loc[student_survey["race"] == 7, "race"] = 15
        student_survey.loc[student_survey["race"] == 1, "race"] = 15
        student_survey['race'] = student_survey['race'] - 10
        student_survey['race'] = student_survey['race'].astype(np.int8)

        student_survey['SES'] = student_survey['X1SES_U']

        self.WEIGHTS = student_survey['W1PARENT']
        # student_survey.drop(['W1PARENT'], axis=1)

        student_survey = student_survey.apply(pd.to_numeric, errors = 'coerce')

        ### Imputation
        # The authors specify that they use ``multiple imputation'' 
        # procedures to impute missing data. Because they do not 
        # specify which procedure they use (beyond that they use t
        # he STATA package), we do best practice work here 
        # (in python) and use an MI library based on LightGBM and 
        # the MICE algorithm.       

        # Set missing values as python NANs
        temp = student_survey.loc[:, student_survey.columns != 'SES']
        temp[temp < 0] = nan
        student_survey.loc[:, student_survey.columns != 'SES'] = temp

        # Create kernel. 
        kds = mf.ImputationKernel(
        student_survey,
        datasets=1,
        save_all_iterations=False,
        random_state=42
        )

        # Run the MICE algorithm for 2 iterations
        kds.mice(2)

        completed_dataset = kds.complete_data(dataset=0, inplace=False)

        # Low teacher support
        low = [i[0] for i in low_teacher_support]
        student_df = completed_dataset[low].dropna()
        teacher_var = (student_df.sum(axis=1)/len(low)).to_frame()
        teacher_var.describe()
        teacher_var.columns = ['teacher']

        # Ability self concepts
        # The scale was reverse-coded so that high scores 
        # signified strong math ability self-concepts 
        # (1 = Strongly disagree, 4 = Strongly agree).
        reverse_code = {
            'S1MTESTS': {1: 4, 2: 3, 3: 2, 4: 1},
            'S1MTEXTBOOK': {1: 4, 2: 3, 3: 2, 4: 1},
            'S1MSKILLS': {1: 4, 2: 3, 3: 2, 4: 1},
            'S1MASSEXCL': {1: 4, 2: 3, 3: 2, 4: 1},
        }
        ability = [i[0] for i in ability_self_concept]
        ability_df = completed_dataset[ability].dropna()
        ability_df = ability_df.replace(reverse_code)
        ability_var = (ability_df.sum(axis=1)/len(ability)).to_frame()
        ability_var.columns = ['ability']

        # Parental support
        parent = [i[0] for i in parental_support]
        parent_df = completed_dataset[parent]
        parental_var = (parent_df.sum(axis=1)/len(parent)).to_frame()
        parental_var.columns = ['parents']

        # Math level 9th grade
        level = [i[0] for i in highest_level_math]
        level_var = completed_dataset[level]
        level_var.columns = ['base_level']

        # Target variable, math acheivement (score) 11th
        acheive = [i[0] for i in math_acheivement_score]
        acheive_var = completed_dataset[acheive]
        acheive_var.columns = ['math']

        # Base math acheivement (score) 9th
        base = [i[0] for i in base_year_score]
        base_var = completed_dataset[base]
        base_var.columns = ['base_math']

        completed_dataset = completed_dataset.replace({'sex': self.SEX_MAP,
                                                       'race': self.RACE_MAP})
                                                        
        # concat full df
        full_df = pd.concat([acheive_var, 
                     teacher_var, 
                     ability_var, 
                     parental_var, 
                     completed_dataset['sex'], 
                     completed_dataset['race'],
                     completed_dataset['SES'], 
                     base_var,
                     level_var,
                     self.WEIGHTS], axis=1)

        full_df.to_pickle(filename)
        return full_df

    def table_2(self):
        # Correlation matrix
        corr_df = self.dataframe[['math', 
                     'teacher', 
                     'ability', 
                     'parents', 
                     'sex', 
                     'SES', 
                     'base_math',
                     'base_level']]
        reverse_sex = {
            "Male": 0,
            "Female": 1
        }
        corr_df = corr_df.replace({'sex':reverse_sex})
        self.corr_df = corr_df.corr()
        return corr_df.corr()
    
    def table_3(self):
        table_3_results = {}

        self.WEIGHTS = self.dataframe['W1PARENT']

        def wls_model(form, name):
            model = WLS.from_formula(
                form,
                data=self.dataframe,
                freq_weights=np.array(self.WEIGHTS.array)
            )
            regression = model.fit(method='pinv')
            table_3_results[name] = regression.summary2()
        
        # Low teacher support (lts) only model
        wls_model(
            'math ~ teacher + C(sex, Treatment(reference="Male")) + C(race, Treatment(reference="White")) + SES + base_math + base_level',
            'model_1'
        )
        # Ability self concepts (sc) only model
        wls_model(
            'math ~ ability + C(sex, Treatment(reference="Male")) + C(race, Treatment(reference="White")) + SES + base_math + base_level',
            'model_2'
        )
        # Interaction: lts x sc
        wls_model(
            'math ~ (teacher * ability) + C(sex, Treatment(reference="Male")) + C(race, Treatment(reference="White")) + SES + base_math + base_level',
            'model_3'
        )
        # Interaction: sc x parental support (ps)
        wls_model(
            'math ~ (ability * parents) + C(sex, Treatment(reference="Male")) + C(race, Treatment(reference="White")) + SES + base_math + base_level',
            'model_4'
        )
        # 3-way Interaction: lts x sc x ps
        wls_model(
            'math ~ (teacher * parents * ability) + C(sex, Treatment(reference="Male")) + C(race, Treatment(reference="White")) + SES + base_math + base_level',
            'model_5'
        )
        self.regression_df = table_3_results
        return table_3_results
    
    def table_2_check(self):
        if self.corr_df is None:
            results = self.table_2()
            self.corr_df = results
        else: 
            results = self.corr_df
        return results
    
    def table_3_check(self):
        if self.regression_df is None:
            results = self.table_3()
            self.regression_df = results
        else: 
            results = self.regression_df
        return results
    
    def figure_1_check(self):
        if self.cov_df is None:
            results = self.figure_1()
            self.cov_df = results
        else: 
            results = self.cov_df
        return results
    
    def figure_1(self):
        # NOTE: Uses the following analysis:
        # http://web.pdx.edu/~newsomj/mlrclass/ho_simple%20slopes.pdf
        # Difficult to replicate, but essentially checks the slopes
        # between high (+1SD) and low(-1SD) values for a given variable
        # which can be found using the covariance matrix.
        model = WLS.from_formula(
                'math ~ (teacher * parents * ability) + C(sex, Treatment(reference="Male")) + C(race, Treatment(reference="White")) + SES + base_math + base_level',
                data=self.dataframe,
                freq_weights=np.array(self.WEIGHTS.array)
        )
        regression = model.fit(method='pinv')
        cov = regression.cov_params()
        self.cov_df = cov
        return cov

    def finding_50_1(self):
        """The analytic sample was 51% female; 
        55% White, 20% Hispanic, 11% Black, 4% Asian, 
        and 9% Other race/ethnicity. A comparison of 
        the analytic sample and the excluded sample is 
        provided in Table 1. Of the 13 comparisons, 5 
        demonstrated at least a small effect; compared to 
        the excluded sample, students in the analytic sample 
        had higher math ability self-concepts in 9th grade 
        (d = 0.21), higher math achievement in 9th grade 
        (d = 0.54) and in 11th grade (d = 0.63). Also, they 
        were more likely to be in more advanced math courses 
        in 9th grade (d = 0.27) and be from families of higher 
        socioeconomic status (d = 0.46)."""
        pass

    def finding_52_1(self):
        """As predicted, a negative correlation was found
        between perceived low math teacher support and 11th 
        grade math achievement (r = -0.11).
        """
        corr_df = self.table_2_check()
        corr_teacher_math = corr_df['math'].loc['teacher']
        soft_finding = (corr_teacher_math < 0.05)
        return ([], soft_finding, [corr_teacher_math])

    def finding_52_2(self):
        """Ability self-concepts and parental support 
        in 9th grade were positively correlated with students' 
        11th grade achievement (r = 0.30 and r = 0.12 respectively).
        """
        corr_df = self.table_2_check()
        corr_ability_math = corr_df['math'].loc['ability']
        corr_parent_math = corr_df['math'].loc['parents']
        soft_finding = (corr_ability_math > 0.2) & (corr_parent_math > 0.05)
        return ([], soft_finding, [corr_ability_math, corr_parent_math])

    def finding_52_3(self):
        """A strong positive correlation was found between 
        9th and 11th grade math achievement (r = 0.74).
        """
        corr_df = self.table_2_check()
        corr_math_math = corr_df['math'].loc['base_math']
        soft_finding = (corr_math_math > 0.5)
        return ([], soft_finding, [corr_math_math])

    def finding_54_1(self):
        """perceived low math teacher support in 11th grade 
        negatively predicted students' 11th grade math achievement 
        (B = -1.51, p &lt; .001) while controlling for students' 
        demographics, 9th grade math achievement score, and math 
        course.
        """
        reg_df = self.table_3_check()
        m = reg_df['model_1']
        B = m.tables[1].loc['teacher']['Coef.']
        soft_finding = (B < -0.5)
        # NOTE: though we do not know how to interpret, p-value
        # is available
        p = m.tables[1].loc['teacher']['P>|t|']
        return ([],soft_finding,[B])

    def finding_54_2(self):
        """Second, as shown under Model 2, math ability self-concepts 
        positively and directly predicted 11th grade math achievement 
        (B = 2.51, p &lt; .001).
        """
        reg_df = self.table_3_check()
        m = reg_df['model_2']
        B = m.tables[1].loc['ability']['Coef.']
        soft_finding = (B > 1.5)
        # NOTE: though we do not know how to interpret, p-value
        # is available
        p = m.tables[1].loc['ability']['P>|t|']
        return ([],soft_finding,[B])

    def finding_54_3(self):
        """Contrary to our hypothesis and as shown under Model 3, 
        the relation between perceived low math teacher support and 
        math achievement was not moderated by adolescents' math 
        ability self-concepts (B = -0.46, p = .329)
        """
        reg_df = self.table_3_check()
        m = reg_df['model_3']
        B = m.tables[1].loc['teacher:ability']['Coef.']
        soft_finding = (B < 0)
        # NOTE: though we do not know how to interpret, p-value
        # is available
        p = m.tables[1].loc['ability']['P>|t|']
        return ([],soft_finding,[B])

    def finding_54_4(self):
        """Third, as hypothesized and shown under Model 5, the 3-way 
        interaction was found to be statistically significant 
        (B = -4.38, p &lt; .05). That is, the interaction between 
        perceived low math teacher support and adolescents' math 
        ability self-concepts in predicting adolescents' math 
        achievement varied by the level of parental support.
        """
        reg_df = self.table_3_check()
        m = reg_df['model_5']
        B = m.tables[1].loc['teacher:parents:ability']['Coef.']
        p = m.tables[1].loc['ability']['P>|t|']
        soft_finding = (B < 0) # & (p < 0.1)
        return ([],soft_finding,[B])

    def finding_54_5(self):
        """As expected, perceived low teacher support was linked 
        to lower achievement when adolescents were low on both 
        protective factors, namely low ability selfconcepts and 
        low parental support (B= -2.23, p = .003).
        """
        # Unclear if this is valid...
        cov = self.figure_1_check()
        B = cov.loc['parents:ability']['teacher']
        soft_finding = (B < -1) # & (p < 0.1)
        return ([],soft_finding,[B])

    def finding_54_6(self):
        """Also as expected, perceived low teacher support was not 
        significantly associated with adolescents' achievement 
        when adolescents with low ability self-concepts received 
        high parental support (B = 0.25, p = .741) or when 
        adolescents with high ability self-concepts had low parental 
        support (B =-1.23, p = .083). That is, adolescents did 
        not significantly differ in their math achievement under 
        high and low perceived math teacher support when they were
        high on one of the protective factors.
        """
        pass

    def finding_54_7(self):
        """In our hypothesis, we expected the negative relation 
        between perceived low math teacher support and adolescents' 
        math achievement to be weaker for adolescents with low math 
        ability self-concepts if they had high parental support 
        compared to low parental support. The slopes of those two 
        groups were significantly different (Fig. 1; t [492] = 
        25.69, p &lt; .001).
        """
        pass

    def finding_54_8(self):
        """Specifically, the association between perceived low 
        teacher support and adolescents' achievement was not 
        statistically significant when adolescents with low 
        ability self-concepts had high parental support but 
        was significantly negative when adolescents lacked 
        both ability self-concepts and parental support.
        """
        pass

    def finding_54_9(self):
        """There was one unexpected finding. Perceived low math 
        teacher support was linked to adolescents' lower math 
        achievement when adolescents were high on both protective 
        factors: high math ability self-concepts and high parental 
        support (B = -1.66, p = .026). We had expected this 
        relation to be non-significant. We should note that this 
        effect was significant at p &lt; .05 with a very large 
        sample size (n = 14,580).
        """
        pass