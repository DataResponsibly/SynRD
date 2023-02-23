from SynRD.publication import Publication, Finding, VisualFinding, FigureFinding

import pandas as pd
import numpy as np

from itertools import chain

from statsmodels.regression.mixed_linear_model import MixedLM

class Pierce2019Who(Publication):
    """
    A class wrapper for all publication classes, for shared functionality.
    """
    DEFAULT_PAPER_ATTRIBUTES = {
        'id': 'pierce2019whoy',
        'length_pages': 21,
        'authors': ['Kayla D. R. Pierce', 'Christopher S. Quiroz'],
        'journal': 'Journal of Social and Personal Relationships',
        'year': 2019,
        'current_citations': 23,
        'base_dataframe_pickle': 'pierce2019who_dataframe.pickle'
    }
    
    SEX_MAP = {
        1: 'male',
        2: 'female'
    }

    EDUCATION_MAP = {
        1: 'less than a high school diploma',
        2: 'high school diploma',
        3: 'some college',
        4: '4-year degree',
        5: 'graduate degree',
    }
    
    DATAFRAME_COLUMNS = ['positive_emotion', 
                         'negative_emotion', 
                         'spouse_support', 
                         'spouse_strain',
                         'child_support', 
                         'child_strain', 
                         'friend_support', 
                         'friend_strain',
                         'confidants', 
                         'age', 
                         'age_category', 
                         'income', 
                         'sex', 
                         'education', 
                         'education_category',
                         'retired', 
                         'num_child']
    
    FILENAME = 'pierce2019who'
    
    table = None
    
    def __init__(self, dataframe=None, filename=None):
        super(Pierce2019Who, self).__init__(dataframe=dataframe)

        self.FINDINGS = self.FINDINGS + [
            VisualFinding(self.table_2, description="table_2"),
            Finding(self.finding_3284_1, description="finding_3284_1",
                    text="""When accounting for between-individual differences, spousal support 
                            has the strongest relationship with positive emotional states, reaffirming 
                            the findings of Walen and Lachman (2000). 
                            """),
            Finding(self.finding_3286_1, description="finding_3286_1",
                    text="""Increased spousal support is associated with an increased positive 
                            emotional state. A direct comparison of the coefficients reveals that 
                            positive spousal support has a 232% greater correlation than support 
                            from children, and a 320% greater correlation than support from friends. 
                            A Wald test comparing coefficients confirms that the correlation stemming 
                            from spousal support is significantly larger than those stemming from children and friends.
                            """),
            Finding(self.finding_3286_2, description="finding_3286_2",
                    text="""the stark difference between support and strain. Support from all three 
                            sources is significantly correlated with more positive emotional states.
                            """),
            Finding(self.finding_3286_3, description="finding_3286_3",
                    text="""However, of the three sources of strain, only the strain stemming from 
                            spouses is significantly correlated with lower positive emotional states. 
                            The other two sources are insignificant predictors of positive emotional 
                            states, meaning that having straining children and friends is not significantly 
                            associated with lower positive emotion.
                            """),
            Finding(self.finding_3286_4, description="finding_3286_4",
                    text="""In the case of positive emotions, only spousal support is shown to have 
                            a significant causal link to positive emotions. That is to say, as spouses 
                            become more supportive over time, individuals report more positive emotional 
                            states.
                            """),
            Finding(self.finding_3286_5, description="finding_3286_5",
                    text="""This was not the case for support from children and friends, despite being 
                            correlated with positive emotions.
                            """),
            Finding(self.finding_3286_6, description="finding_3286_6",
                    text="""In the case of social strain, none of the within-individual metrics are 
                            significant predictors of positive emotions. This finding demonstrates that 
                            as spouses, children, and friends exerted more strain, there is no significant 
                            change on the reported level of positive emotions.
                            """),
            Finding(self.finding_3286_7, description="finding_3286_7",
                    text="""Similar to the results for positive emotional states, we found that spouses 
                            have the greatest overall correlation with negative emotional states.
                            """),
            Finding(self.finding_3286_8, description="finding_3286_8",
                    text="""Spousal support and friend support are the only types of support to be 
                            negatively correlated with negative emotional states.
                            """),
            Finding(self.finding_3286_9, description="finding_3286_9",
                    text="""Furthermore, the correlation between spousal support and negative emotional 
                            states is 244% greater than the correlation of support stemming from friends. 
                            A Wald test confirms that the difference between spousal support and friend 
                            support is statistically significant.
                            """),
            Finding(self.finding_3286_10, description="finding_3286_10",
                    text="""For the between-individual coefficients regarding strain, we found significant 
                            correlations between strain and negative emotions stemming from both spouses 
                            and children. Although the coefficient for spousal strain is greater in magnitude
                            than that of child strain, the Wald test comparing the coefficients demonstrates 
                            that there is no significant difference between the measures.
                            """),
            Finding(self.finding_3286_11, description="finding_3286_11",
                    text="""Finally, we find no significant correlation between friend-based strain and 
                            negative emotional states.
                            """),
            Finding(self.finding_3287_1, description="finding_3287_1",
                    text="""Similar to positive emotional states, both support and strain from spouses 
                            have a significant causal link to individual’s negative emotional states in 
                            the predicted directions.
                            """),
            Finding(self.finding_3287_2, description="finding_3287_2",
                    text="""However, the results for negative emotional states differ slightly from positive 
                            emotions in that child-based strain is also a significant causal factor. 
                            The magnitude of child-based strain is similar in size to that of spouse-based 
                            strain.
                            """),
            Finding(self.finding_3287_3, description="finding_3287_3",
                    text="""Unsurprisingly, the results show that individuals who earn high incomes are 
                            more likely to report positive emotional states and less likely to report negative 
                            emotional states than their poorer counterparts.
                            """),
            Finding(self.finding_3287_4, description="finding_3287_4",
                    text="""Interestingly, this pattern is not observed at the within-person level. That 
                            is to say, as the same individual earns a greater amount money throughout her 
                            life, she is no more likely to report positive emotional states than when she 
                            was earning less money.
                            """),
            Finding(self.finding_3287_5, description="finding_3287_5",
                    text="""The same null finding holds true for negative emotional states.
                            """),
            Finding(self.finding_3287_6, description="finding_3287_6",
                    text="""we find no relationship between chronological age and positive emotional states, 
                            but we do find a nonlinear correlation between age and negative emotional states.
                            """),
            Finding(self.finding_3287_7, description="finding_3287_7",
                    text="""Based on the coefficients, we know that as people age, they are more likely to 
                            report negative emotional states. This trend reverses at the apex of the curve 
                            which we calculate to be approximately 65 years of age.
                            """),
            Finding(self.finding_3287_8, description="finding_3287_8",
                    text="""However, it is important to note that 90% of the responses in our data are from 
                            people at 34–74 years of age, thus leading to the conclusion that the nonlinear 
                            pattern is asymmetric.
                            """),
            Finding(self.finding_3287_9, description="finding_3287_9",
                    text="""Although gender does not significantly correlate with positive emotional states, 
                            we find that it does significantly correlate to negative emotional states.
                            """),
            Finding(self.finding_3287_10, description="finding_3287_10",
                    text="""All else equal, men are in fact less likely to report negative emotional states 
                            than their female counterparts. To put the magnitude of the gender effect into 
                            context, being a man has the same benefit as making US$5,500 more per year on 
                            negative emotional outcomes.
                            """),
        ]
        
    def _recreate_dataframe(self, filename='pierce2019who_dataframe.pickle'):
        raw = pd.read_csv('./data/DS0001/04690-0001-Data.tsv',sep='\t')
        df = raw[(raw['V2060'] == 1) & (raw['V2225'] > 0) & (raw['V2017'] > 0)]
        
        missing_val = {-95: np.nan, -96: np.nan, -99: np.nan}
        # check_box = {1: 1, 5: 0}
        df.replace({'V103': missing_val, 'V104': missing_val, 'V546': missing_val,
                    'V1002': missing_val, 'V1006': missing_val, 'V1007': missing_val,
                    'V1010': missing_val, 'V1012': missing_val, 'V2007': missing_val,
                    'V2017':missing_val, 'V2020': missing_val,
                    }, inplace=True)
        
        # Listwise deletion as in the original paper
        df = df.dropna(axis='index', subset=['V1002','V1006','V1007','V1010','V1012'], how='any')

        # Dependent variables
        df['positive_emotion'] = df[['V1006', 'V1010']].mean(axis=1)
        df['negative_emotion'] = df[['V1012', 'V1002', 'V1007']].mean(axis=1)

        # Normalize
        df['positive_emotion']=(df['positive_emotion']-df['positive_emotion'].min()) \
                                /(df['positive_emotion'].max()-df['positive_emotion'].min())
        df['negative_emotion']=(df['negative_emotion']-df['negative_emotion'].min()) \
                                /(df['negative_emotion'].max()-df['negative_emotion'].min())
        
        # Independent variables
        df['spouse_support'] = df['V2204']
        df['spouse_strain'] = df['V2205']
        df['child_support'] = df['V2207']
        df['child_strain'] = df['V2208']
        df['friend_support'] = df['V2216']
        df['friend_strain'] = df['V2217']
        
        def age_categorize(row):  
            if row['age'] < 45:
                return '0'
            elif row['age'] >= 45 and row['age'] <= 65:
                return '1'
            elif row['age'] > 65:
                return '2'
            
        def education_categorize(row):  
            if row['education'] < 12:
                return '0'
            elif 12 <= row['education'] < 14:
                return '1'
            elif 14 <= row['education'] < 16:
                return '2'
            elif 16 <= row['education'] < 17:
                return '3'
            elif row['education'] >= 17:
                return '4'
        
        # Control variables
        df['confidants'] = df['V546']
        df['age'] = df['V104']
        df['income'] = df['V2020']
        df['sex'] = df['V103']
        df['education'] = df['V2007']
        df['retired'] = df['V1105']
        df['num_child'] = df['V2017']

        df = df.dropna(axis='index', subset=['confidants','age','age','income','sex',
                                            'education','retired','num_child'], how='any')
        df['age_category'] = df.apply(lambda row: age_categorize(row), axis=1)
        df['education_category'] = df.apply(lambda row: education_categorize(row), axis=1)
        df = df[self.DATAFRAME_COLUMNS]
        df.to_pickle(filename)
        
        return df
    
    def table_2(self):
        table_2_results = {}

        df_lm = self.dataframe.replace({'sex': self.SEX_MAP,
                                         'education_category': self.EDUCATION_MAP,})
        
        vc = {'confidants': '0 + C(confidants)', 'age_category': '0 + C(age_category)', 'education_category': '0 + C(education_category)', 
            'income': '0 + C(income)', 'sex': '0 + C(sex)', 'retired': '0 + C(retired)', 'num_child': '0 + C(num_child)'} 

        def mlm_model(form, name):
            model = MixedLM.from_formula(
                form, 
                vc_formula=vc, 
                data=df_lm, 
                groups=df_lm['age_category']
            )
            result = model.fit()
            table_2_results[name] = result.summary2()
            
        # TODO: Modify stats equations
        # Positive emotion model
        mlm_model(
            'positive_emotion ~ spouse_support + spouse_strain + child_support + child_strain + friend_support + friend_strain \
                + C(confidants) + C(sex) + C(income) + C(education_category) + C(age_category) + C(retired) + C(num_child)',
            'positive_model'
        )
        
        # Negative emotion model
        mlm_model(
            'negative_emotion ~ spouse_support + spouse_strain + child_support + child_strain + friend_support + friend_strain \
                + C(confidants) + C(sex) + C(income) + C(education_category) + C(age_category) + C(retired) + C(num_child)',
            'negative_model'
        )
    
        self.table = table_2_results
        return table_2_results
    
    def table_2_check(self):
        if self.table is None:
            results = self.table_2()
            self.table = results
        else: 
            results = self.table
        return results
    
    def finding_3284_1(self):
        """
        When accounting for between-individual differences, spousal support 
        has the strongest relationship with positive emotional states, reaffirming 
        the findings of Walen and Lachman (2000). 
        """
        df = self.table_2_check()
        pos = df['positive_model']
        spouse = pos.tables[1].loc['spouse_support']['Coef.']
        child = pos.tables[1].loc['child_support']['Coef.']
        friends = pos.tables[1].loc['friend_support']['Coef.']
        soft_finding = (spouse > child) and (spouse > friends)
        return soft_finding
    
    def finding_3286_1(self):
        """
        Increased spousal support is associated with an increased positive 
        emotional state. A direct comparison of the coefficients reveals that 
        positive spousal support has a 232% greater correlation than support 
        from children, and a 320% greater correlation than support from friends. 
        A Wald test comparing coefficients confirms that the correlation stemming 
        from spousal support is significantly larger than those stemming from children and friends.
        """
        df = self.table_2_check()
        pos = df['positive_model']
        spouse = pos.tables[1].loc['spouse_support']['Coef.']
        child = pos.tables[1].loc['child_support']['Coef.']
        friends = pos.tables[1].loc['friend_support']['Coef.']
        soft_finding = (spouse == child * 2.32) and (spouse == friends * 3.2)
        return soft_finding
    
    def finding_3286_2(self):
        """
        the stark difference between support and strain. Support from all three 
        sources is significantly correlated with more positive emotional states.
        """
        df = self.table_2_check()
        pos = df['positive_model']
        spouse = pos.tables[1].loc['spouse_support']['Coef.']
        child = pos.tables[1].loc['child_support']['Coef.']
        friends = pos.tables[1].loc['friend_support']['Coef.']
        soft_finding = (spouse > 0) and (child > 0) and (friends > 0)
        return soft_finding
    
    def finding_3286_3(self):
        """
        However, of the three sources of strain, only the strain stemming from 
        spouses is significantly correlated with lower positive emotional states. 
        The other two sources are insignificant predictors of positive emotional 
        states, meaning that having straining children and friends is not significantly 
        associated with lower positive emotion.
        """
        df = self.table_2_check()
        pos = df['positive_model']
        spouse = pos.tables[1].loc['spouse_strain']['Coef.']
        child = pos.tables[1].loc['child_strain']['Coef.']
        friends = pos.tables[1].loc['friend_strain']['Coef.']
        soft_finding = (spouse < 0) and (child > 0) and (friends > 0)
        return soft_finding
    
    def finding_3286_4(self):
        """
        In the case of positive emotions, only spousal support is shown to have 
        a significant causal link to positive emotions. That is to say, as spouses 
        become more supportive over time, individuals report more positive emotional 
        states.
        """
        df = self.table_2_check()
        pos = df['positive_model']
        spouse = pos.tables[1].loc['spouse_support']['Coef.']
        child = pos.tables[1].loc['child_support']['Coef.']
        friends = pos.tables[1].loc['friend_support']['Coef.']
        soft_finding = (spouse > child) and (spouse > friends)
        return soft_finding
    
    def finding_3286_5(self):
        """
        This was not the case for support from children and friends, despite being 
        correlated with positive emotions.
        """
        df = self.table_2_check()
        pos = df['positive_model']
        spouse = pos.tables[1].loc['spouse_support']['Coef.']
        child = pos.tables[1].loc['child_support']['Coef.']
        friends = pos.tables[1].loc['friend_support']['Coef.']
        soft_finding = (child > 0) and (friends > 0)
        return soft_finding
    
    def finding_3286_6(self):
        """
        In the case of social strain, none of the within-individual metrics are 
        significant predictors of positive emotions. This finding demonstrates that 
        as spouses, children, and friends exerted more strain, there is no significant 
        change on the reported level of positive emotions.
        """
        pass
    
    def finding_3286_7(self):
        """
        Similar to the results for positive emotional states, we found that spouses 
        have the greatest overall correlation with negative emotional states.
        """
        df = self.table_2_check()
        neg = df['negative_model']
        spouse = neg.tables[1].loc['spouse_strain']['Coef.']
        child = neg.tables[1].loc['child_strain']['Coef.']
        friends = neg.tables[1].loc['friend_strain']['Coef.']
        soft_finding = (spouse > child) and (spouse > friends)
        return soft_finding
        
    def finding_3286_8(self):
        """
        Spousal support and friend support are the only types of support to be 
        negatively correlated with negative emotional states.
        """
        df = self.table_2_check()
        neg = df['negative_model']
        spouse = neg.tables[1].loc['spouse_support']['Coef.']
        friends = neg.tables[1].loc['friend_support']['Coef.']
        soft_finding = (spouse < 0) and (friends < 0)
        return soft_finding
    
    def finding_3286_9(self):
        """
        Furthermore, the correlation between spousal support and negative emotional 
        states is 244% greater than the correlation of support stemming from friends. 
        A Wald test confirms that the difference between spousal support and friend 
        support is statistically significant.
        """
        df = self.table_2_check()
        neg = df['negative_model']
        spouse = neg.tables[1].loc['spouse_support']['Coef.']
        friends = neg.tables[1].loc['friend_support']['Coef.']
        soft_finding = (spouse == friends * 2.44)
        return soft_finding
    
    def finding_3286_10(self):
        """
        For the between-individual coefficients regarding strain, we found significant 
        correlations between strain and negative emotions stemming from both spouses 
        and children. Although the coefficient for spousal strain is greater in magnitude
        than that of child strain, the Wald test comparing the coefficients demonstrates 
        that there is no significant difference between the measures.
        """
        df = self.table_2_check()
        neg = df['negative_model']
        spouse = neg.tables[1].loc['spouse_strain']['Coef.']
        child = neg.tables[1].loc['child_strain']['Coef.']
        soft_finding = (spouse >= child)
        return soft_finding
    
    def finding_3286_11(self):
        """
        Finally, we find no significant correlation between friend-based strain and 
        negative emotional states.
        """
        df = self.table_2_check()
        neg = df['negative_model']
        friends = neg.tables[1].loc['friend_strain']['Coef.']
        soft_finding = (friends >= 0)
        return soft_finding
    
    def finding_3287_1(self):
        """
        Similar to positive emotional states, both support and strain from spouses 
        have a significant causal link to individual’s negative emotional states in 
        the predicted directions.
        """
        df = self.table_2_check()
        neg = df['negative_model']
        support = neg.tables[1].loc['spouse_support']['Coef.']
        strain = neg.tables[1].loc['spouse_strain']['Coef.']
        soft_finding = (support > 0) and (strain > 0)
        return soft_finding
    
    def finding_3287_2(self):
        """
        However, the results for negative emotional states differ slightly from positive 
        emotions in that child-based strain is also a significant causal factor. 
        The magnitude of child-based strain is similar in size to that of spouse-based 
        strain.
        """
        df = self.table_2_check()
        neg = df['negative_model']
        child = neg.tables[1].loc['child_strain']['Coef.']
        soft_finding = (child > 0)
        return soft_finding 
    
    def finding_3287_3(self):
        """
        Unsurprisingly, the results show that individuals who earn high incomes are 
        more likely to report positive emotional states and less likely to report negative 
        emotional states than their poorer counterparts.
        """
        df = self.table_2_check()
        pos = df['positive_model']
        neg = df['negative_model']
        high_income = pos.tables[1].loc['income']['Coef.']
        low_income = neg.tables[1].loc['income']['Coef.']
        soft_finding = (high_income > 0) and (low_income < 0)
        return soft_finding
    
    def finding_3287_4(self):
        """
        Interestingly, this pattern is not observed at the within-person level. That 
        is to say, as the same individual earns a greater amount money throughout her 
        life, she is no more likely to report positive emotional states than when she 
        was earning less money.
        """
        # TODO: Within variable
        pass
    
    def finding_3287_5(self):
        """
        The same null finding holds true for negative emotional states.
        """
        # TODO: Within variable
        pass
        
    
    def finding_3287_6(self):
        """
        we find no relationship between chronological age and positive emotional states, 
        but we do find a nonlinear correlation between age and negative emotional states.
        """
        df = self.table_2_check()
        neg = df['negative_model']
        age = neg.tables[1].loc['age']['Coef.']
        soft_finding = (age < 0)
        return soft_finding
    
    def finding_3287_7(self):
        """
        Based on the coefficients, we know that as people age, they are more likely to 
        report negative emotional states. This trend reverses at the apex of the curve 
        which we calculate to be approximately 65 years of age.
        """
        # TODO: Age Categories
        pass
    
    def finding_3287_8(self):
        """
        However, it is important to note that 90% of the responses in our data are from 
        people at 34–74 years of age, thus leading to the conclusion that the nonlinear 
        pattern is asymmetric.
        """
        # TODO: Age Categories
        pass
    
    def finding_3287_9(self):
        """
        Although gender does not significantly correlate with positive emotional states, 
        we find that it does significantly correlate to negative emotional states.
        """
        df = self.table_2_check()
        neg = df['negative_model']
        sex = neg.tables[1].loc['sex']['Coef.']
        soft_finding = (sex < 0)
        return soft_finding
        
    
    def finding_3287_10(self):
        """
        All else equal, men are in fact less likely to report negative emotional states 
        than their female counterparts. To put the magnitude of the gender effect into 
        context, being a man has the same benefit as making US$5,500 more per year on 
        negative emotional outcomes.
        """
        # TODO: Income
        pass
    
    