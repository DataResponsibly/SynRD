from meta_classes import Publication, Finding, VisualFinding, FigureFinding

import pandas as pd
import numpy as np

from itertools import chain

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
                         'age_group', 
                         'income', 
                         'sex', 
                         'education', 
                         'education_group',
                         'retired', 
                         'num_child']
    
    FILENAME = 'pierce2019who'
    
    
    def __init__(self, dataframe=None, filename=None):
        if filename is not None:
            self.dataframe = pd.read_pickle(filename)
        elif dataframe is not None:
            self.dataframe = dataframe
        else:
            self.dataframe = self._recreate_dataframe()

        self.FINDINGS = self.FINDINGS + [
            # VisualFinding(self.table_b2, description="table_b2"),
            # FigureFinding(self.figure_2, description="figure_2"),
            Finding(self.finding_3284_1, description="finding_3284_1",
                    text="""When accounting for between-individual differences, spousal support 
                            has the strongest relationship with positive emotional states, reaffirming 
                            the findings of Walen and Lachman (2000). Increased spousal support 
                            is associated with an increased positive emotional state. A direct 
                            comparison of the coefficients reveals that positive spousal support 
                            has a 232% greater correlation than support from children, and a 320% 
                            greater correlation than support from friends. A Wald test comparing 
                            coefficients confirms that the correlation stemming from spousal support 
                            is significantly larger than those stemming from children and friends.
                            """),
            Finding(self.finding_3286_1, description="finding_3286_1",
                    text="""the stark difference between support and strain. Support from all three 
                            sources is significantly correlated with more positive emotional states.
                            """),
            Finding(self.finding_3286_2, description="finding_3286_2",
                    text="""However, of the three sources of strain, only the strain stemming from 
                            spouses is significantly correlated with lower positive emotional states. 
                            The other two sources are insignificant predictors of positive emotional 
                            states, meaning that having straining children and friends is not significantly 
                            associated with lower positive emotion.
                            """),
            Finding(self.finding_3286_3, description="finding_3286_3",
                    text="""In the case of positive emotions, only spousal support is shown to have 
                            a significant causal link to positive emotions. That is to say, as spouses 
                            become more supportive over time, individuals report more positive emotional 
                            states.
                            """),
            Finding(self.finding_3286_4, description="finding_3286_4",
                    text="""This was not the case for support from children and friends, despite being 
                            correlated with positive emotions.
                            """),
            Finding(self.finding_3286_5, description="finding_3286_5",
                    text="""In the case of social strain, none of the within-individual metrics are 
                            significant predictors of positive emotions. This finding demonstrates that 
                            as spouses, children, and friends exerted more strain, there is no significant 
                            change on the reported level of positive emotions.
                            """),
            Finding(self.finding_3286_6, description="finding_3286_6",
                    text="""Similar to the results for positive emotional states, we found that spouses 
                            have the greatest overall correlation with negative emotional states.
                            """),
            Finding(self.finding_3286_7, description="finding_3286_7",
                    text="""Spousal support and friend support are the only types of support to be 
                            negatively correlated with negative emotional states.
                            """),
            Finding(self.finding_3286_8, description="finding_3286_8",
                    text="""Furthermore, the correlation between spousal support and negative emotional 
                            states is 244% greater than the correlation of support stemming from friends. 
                            A Wald test confirms that the difference between spousal support and friend 
                            support is statistically significant.
                            """),
            Finding(self.finding_3286_9, description="finding_3286_9",
                    text="""For the between-individual coefficients regarding strain, we found significant 
                            correlations between strain and negative emotions stemming from both spouses 
                            and children. Although the coefficient for spousal strain is greater in magnitude
                            than that of child strain, the Wald test comparing the coefficients demonstrates 
                            that there is no significant difference between the measures.
                            """),
            Finding(self.finding_3286_10, description="finding_3286_10",
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
        check_box = {1: 1, 5: 0}
        df.replace({'V103': missing_val, 'V104': missing_val, 'V546': missing_val,
                    'V1002': missing_val, 'V1006': missing_val, 'V1007': missing_val,
                    'V1010': missing_val, 'V1012': missing_val, 'V2007': missing_val,
                    'V2017':missing_val, 'V2020': missing_val,
                    'V1105': check_box,
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
                return 'under 45'
            elif row['age'] >= 45 and row['age'] <= 65:
                return '45-65'
            elif row['age'] > 65:
                return 'over 65'
            
        def education_categorize(row):  
            if row['education'] < 12:
                return '1'
            elif 12 <= row['education'] < 14:
                return '2'
            elif 14 <= row['education'] < 16:
                return '3'
            elif 16 <= row['education'] < 17:
                return '4'
            elif row['education'] >= 17:
                return '5'
        
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

        df['age_group'] = df.apply(lambda row: age_categorize(row), axis=1)
        df['education_group'] = df.apply(lambda row: education_categorize(row), axis=1)
        df = df[self.DATAFRAME_COLUMNS]
        df.to_pickle(filename)
        
        return df
    
    
    def finding_3284_1(self):
        """
        When accounting for between-individual differences, spousal support 
        has the strongest relationship with positive emotional states, reaffirming 
        the findings of Walen and Lachman (2000). Increased spousal support 
        is associated with an increased positive emotional state. A direct 
        comparison of the coefficients reveals that positive spousal support 
        has a 232% greater correlation than support from children, and a 320% 
        greater correlation than support from friends. A Wald test comparing 
        coefficients confirms that the correlation stemming from spousal support 
        is significantly larger than those stemming from children and friends.
        """
        pass
    
    
    def finding_3286_1(self):
        """
        the stark difference between support and strain. Support from all three 
        sources is significantly correlated with more positive emotional states.
        """
        pass
    
    
    def finding_3286_2(self):
        """
        However, of the three sources of strain, only the strain stemming from 
        spouses is significantly correlated with lower positive emotional states. 
        The other two sources are insignificant predictors of positive emotional 
        states, meaning that having straining children and friends is not significantly 
        associated with lower positive emotion.
        """
        pass

    
    def finding_3286_3(self):
        """
        In the case of positive emotions, only spousal support is shown to have 
        a significant causal link to positive emotions. That is to say, as spouses 
        become more supportive over time, individuals report more positive emotional 
        states.
        """
        pass
    
    def finding_3286_4(self):
        """
        This was not the case for support from children and friends, despite being 
        correlated with positive emotions.
        """
        pass
    
    def finding_3286_5(self):
        """
        In the case of social strain, none of the within-individual metrics are 
        significant predictors of positive emotions. This finding demonstrates that 
        as spouses, children, and friends exerted more strain, there is no significant 
        change on the reported level of positive emotions.
        """
        pass
    
    def finding_3286_6(self):
        """
        Similar to the results for positive emotional states, we found that spouses 
        have the greatest overall correlation with negative emotional states.
        """
        pass
    
    def finding_3286_7(self):
        """
        Spousal support and friend support are the only types of support to be 
        negatively correlated with negative emotional states.
        """
        pass
    
    def finding_3286_8(self):
        """
        Furthermore, the correlation between spousal support and negative emotional 
        states is 244% greater than the correlation of support stemming from friends. 
        A Wald test confirms that the difference between spousal support and friend 
        support is statistically significant.
        """
        pass
    
    def finding_3286_9(self):
        """
        For the between-individual coefficients regarding strain, we found significant 
        correlations between strain and negative emotions stemming from both spouses 
        and children. Although the coefficient for spousal strain is greater in magnitude
        than that of child strain, the Wald test comparing the coefficients demonstrates 
        that there is no significant difference between the measures.
        """
        pass
    
    def finding_3286_10(self):
        """
        Finally, we find no significant correlation between friend-based strain and 
        negative emotional states.
        """
        pass
    
    def finding_3287_1(self):
        """
        Similar to positive emotional states, both support and strain from spouses 
        have a significant causal link to individual’s negative emotional states in 
        the predicted directions.
        """
        pass
    
    def finding_3287_2(self):
        """
        However, the results for negative emotional states differ slightly from positive 
        emotions in that child-based strain is also a significant causal factor. 
        The magnitude of child-based strain is similar in size to that of spouse-based 
        strain.
        """
        pass
    
    def finding_3287_3(self):
        """
        Unsurprisingly, the results show that individuals who earn high incomes are 
        more likely to report positive emotional states and less likely to report negative 
        emotional states than their poorer counterparts.
        """
        pass
    
    def finding_3287_4(self):
        """
        Interestingly, this pattern is not observed at the within-person level. That 
        is to say, as the same individual earns a greater amount money throughout her 
        life, she is no more likely to report positive emotional states than when she 
        was earning less money.
        """
        pass
    
    def finding_3287_5(self):
        """
        The same null finding holds true for negative emotional states.
        """
        pass
    
    def finding_3287_6(self):
        """
        we find no relationship between chronological age and positive emotional states, 
        but we do find a nonlinear correlation between age and negative emotional states.
        """
        pass
    
    def finding_3287_7(self):
        """
        Based on the coefficients, we know that as people age, they are more likely to 
        report negative emotional states. This trend reverses at the apex of the curve 
        which we calculate to be approximately 65 years of age.
        """
        pass
    
    def finding_3287_8(self):
        """
        However, it is important to note that 90% of the responses in our data are from 
        people at 34–74 years of age, thus leading to the conclusion that the nonlinear 
        pattern is asymmetric.
        """
        pass
    
    def finding_3287_9(self):
        """
        Although gender does not significantly correlate with positive emotional states, 
        we find that it does significantly correlate to negative emotional states.
        """
        pass
    
    def finding_3287_1(self):
        """
        All else equal, men are in fact less likely to report negative emotional states 
        than their female counterparts. To put the magnitude of the gender effect into 
        context, being a man has the same benefit as making US$5,500 more per year on 
        negative emotional outcomes.
        """
        pass
    
    