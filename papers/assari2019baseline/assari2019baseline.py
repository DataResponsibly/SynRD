
import math
from SynRD.publication import Publication, Finding, VisualFinding, TAXONOMY

import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.stats.weightstats

class Assari2019Baseline(Publication):
    DEFAULT_PAPER_ATTRIBUTES = {
        'id': 'assari2019baseline',
        'length_pages': 15,
        'authors': ['Shervin Assari', 'Mohsen Bazargan'],
        'journal': 'International Journal of Environmental Research and Public Health',
        'year': 2019,
        'current_citations': 9, #number of citations the paper has or how many people have cited it?
        'base_dataframe_pickle': 'assari2019ability_dataframe.pickle'
    }

    RACE_MAP = {
        1: "White",
        2: "Black"
    }

    GENDER_MAP = {
        1: "Man",
        2: "Woman"
    }

    FILENAME = 'assari2019baseline'

    COLUMN_MAP =  {"V2102": "Race", "V103": "Gender", "V2000": "Age", "V2007": "Education", "V2020": "Income", "V2637": "Smoking", "V2623": "BMI", "V2681": "HTN", "V13214": "Exercise", "V2203": "Depressive symptoms", "V915": "Health", "V1860": "Weight", "V15003": "Response pattern", "V836": "Stroke wave 1", "V4838": "Stroke wave 2", "V10225": "Stroke wave 3", "V12305": "Stroke wave 4", "V15944": "Stroke wave 5", "V12302": "Any stroke"}

    corr_df = None
    means = None
    dead = None

    def __init__(self, dataframe=None):
        super(Assari2019Baseline, self).__init__(dataframe=dataframe)

        self.FINDINGS = self.FINDINGS + [
            Finding(self.finding_5_1, description="finding_5_1",
                    text="""Blacks were younger, had higher number of chronic medical conditions at baseline in comparison to Whites."""),
            Finding(self.finding_5_2, description="finding_5_2",
                    text="""Relative to White people, Black individuals had also lower educational attainment (p < 0.05 for all)."""),
            Finding(self.finding_5_3, description="finding_5_3",
                    text="""Blacks also reported worse self-rated health (SRH) than Whites (Table 1)."""),
            Finding(self.finding_5_6, description="finding_5_6",
                    text="""Similarly, overall, people had 12.53 years of schooling at baseline (95%CI = 12.34-12.73)."""),
            Finding(self.finding_5_7, description="finding_5_7",
                    text="""A comparison of racial groups showed higher educational attainment in Whites (12.69, 95%CI=12.48-12.90) than Blacks (11.37,95%CI = 10.90-11.84). Thus, on average, Whites had more than 1.3 years higher years [sic] of schooling than Blacks..."""),
            Finding(self.finding_5_8, description="finding_5_8",
                    text="""Of the 177 that died, 121 were White (68.36%) and 56 were Black (31.64%)."""),
            Finding(self.finding_5_9, description="finding_5_9",
                    text="""Of the 177 that died, 33 were obese (18.64%) and 144 were not obese (81.36%) at baseline."""),
            Finding(self.finding_6_1, description="finding_6_1",
                    text="""In bivariate association, race was not associated with death due to cerebrovascular (unadjusted HR for Blacks compared to Whites = 0.78, 95% CI = 0.55-1.11), suggesting that Whites and Blacks had similar risk of future cerebrovascular mortality over 25 years."""),
            Finding(self.finding_6_2, description="finding_6_2",
                    text="""In bivariate association, baseline obesity was not associated with future risk of cerebrovascular mortality (Unadjusted HR for Blacks compared to Whites = 0.84, 95% CI = 0.45-1.56), suggesting that Whites and Blacks had a similar risk of future cerebrovascular mortality over 25 years."""),
            Finding(self.finding_6_3, description="finding_6_3",
                    text="""Race (Black) was negatively associated with education and income"""),
            Finding(self.finding_6_4, description="finding_6_4",
                    text="""[race (Black) was]... positively associated with depressive symptoms, hypertension, and obesity."""),
            Finding(self.finding_6_5, description="finding_6_5",
                    text="""Blacks more frequently smoked and less frequently exercised."""),
            Finding(self.finding_6_6, description="finding_6_6",
                    text="""Race was not associated with cerebrovascular death."""),
            Finding(self.finding_6_7, description="finding_6_7",
                    text="""Baseline obesity was associated with female gender and less education, income, smoking, and exercise."""),
            Finding(self.finding_6_8, description="finding_6_8",
                    text="""Obesity at baseline was associated with depressive symptoms and hypertension at baseline."""),
            Finding(self.finding_6_9, description="finding_6_9",
                    text="""Obesity at baseline was not associated with cerebrovascular death in the pooled sample (Table 2).""")
        ]    
    
    def _get_any_stroke_if_died(self, x):
        response_pattern = str(x["Response pattern"])
        if "4" not in response_pattern:
            return 0 # patient did not die
        for i in range(5):
            if x[f"Stroke wave {i + 1}"] == 1:
                return 1
        return 0
    
    def _recreate_dataframe(self, filename='assari2019baseline_dataframe.pickle'):
        data = pd.read_csv('data/DS0001/04690-0001-Data.tsv', sep='\t')

        data = data[self.COLUMN_MAP.keys()]
        data.rename(columns=self.COLUMN_MAP, inplace=True)

        data = data[(data["Race"] == 1) | (data["Race"] == 2)] # 1 = white, 2 = Black
        data["Educational attainment"] = data.apply(lambda x: 1 if x["Education"] >= 12 else 0, axis=1)
        data["Obesity"] = data.apply(lambda x: 1 if x["BMI"] > 30 else 0, axis=1)
        data["Health binary"] = data.apply(lambda x: 1 if x["Health"] in [1, 2, 3] else 0, axis=1)
        data["Death to cerebrovascular disease"] = data.apply(lambda x: self._get_any_stroke_if_died(x), axis=1)
        data.drop(columns=['Stroke wave 1', 'Stroke wave 2', 'Stroke wave 3', 'Stroke wave 4', 'Stroke wave 5','Response pattern', 'Any stroke'], inplace=True)

        data.to_pickle(filename)
        return data
    
    def get_corr(self):
        if self.corr_df is None:
            corr_df = self.dataframe[['Race', 'Age', 'Gender', 'Education', 'Income', 'Smoking', 'Exercise', 'Depressive symptoms', 'HTN', 'Obesity', 'Death to cerebrovascular disease']]
            self.corr_df = corr_df.corr()
        return self.corr_df
    
    def get_race_pools_with_means(self):
        if self.means is None:
            black_pool = self.dataframe.loc[self.dataframe['Race'] == 2]
            white_pool = self.dataframe.loc[self.dataframe['Race'] == 1]

            black_pool_means, white_pool_means = self._get_adjusted_means(black_pool), self._get_adjusted_means(white_pool)
            means = pd.concat([black_pool_means, white_pool_means])
            means['Race'] = ['Black', 'White']
            means.set_index('Race', inplace=True)

            self.means = means
        return self.means
    
    def _get_adjusted_means(self, data_sample):
        temp_means = np.around(sm.stats.weightstats.DescrStatsW(data_sample, weights=data_sample['Weight']).mean, 4)
        return pd.DataFrame(data=[temp_means], columns=data_sample.columns)
    
    def get_dead(self):
        if self.dead is None:
            self.dead = self.dataframe.loc[self.dataframe['Death to cerebrovascular disease'] == 1]
        return self.dead

    def finding_5_1(self):
        """Blacks were younger, had higher number of chronic medical conditions at baseline in comparison to Whites."""
        means = self.get_race_pools_with_means()
        black_age = means['Age']['Black']
        white_age = means['Age']['White']
        black_htn = means['HTN']['Black']
        white_htn = means['HTN']['White']
        values = [black_age, white_age, black_htn, white_htn]
        soft_finding = black_age < white_age and black_htn > white_htn
        return (values, soft_finding, values)

    def finding_5_2(self):
        """Relative to White people, Black individuals had also lower educational attainment (p < 0.05 for all)."""
        means = self.get_race_pools_with_means()
        black_education = means['Education']['Black']
        white_education = means['Education']['White']
        values = [black_education, white_education]
        soft_finding =  black_education < white_education
        return (values, soft_finding, values)

    def finding_5_3(self):
        """Blacks also reported worse self-rated health (SRH) than Whites (Table 1)."""
        means = self.get_race_pools_with_means()
        black_health = means['Health']['Black']
        white_health = means['Health']['White']
        values = [black_health, white_health]
        soft_finding = black_health > white_health # note 1 = excellent, 5 = poor
        return (values, soft_finding, values)

    def finding_5_4(self):
        """The overall prevalence of DM was 5.73%, (95%CI = 4.80-6.82)."""
        pass
    
    def finding_5_5(self):
        """DM was more common in Blacks (9.22%, 95%CI = 7.75-10.95) than Whites (5.25%, 95%CI = 4.2.4-6.50)."""
        pass

    def finding_5_6(self):
        """Similarly, overall, people had 12.53 years of schooling at baseline (95%CI = 12.34-12.73)."""
        means = self._get_adjusted_means(self.dataframe)
        years_schooling = means['Education'][0]
        soft_finding = round(years_schooling, 2) == 12.53
        return ([years_schooling], soft_finding, [years_schooling])

    def finding_5_7(self):
        """A comparison of racial groups showed higher educational attainment in Whites (12.69, 95%CI=12.48-12.90) than Blacks (11.37,95%CI = 10.90-11.84). Thus, on average, Whites had more than 1.3 years higher years [sic] of schooling than Blacks..."""
        means = self.get_race_pools_with_means()
        white_education = means['Education']['White']
        black_education = means['Education']['Black']
        values = [white_education, black_education]
        soft_finding = white_education > black_education + 1.3
        return (values, soft_finding, values)

    def finding_5_8(self):
        """Of the 177 that died, 121 were White (68.36%) and 56 were Black (31.64%)."""
        dead = self.get_dead()
        total = dead.shape[0]
        black_count = dead.loc[dead['Race'] == 2].shape[0]
        white_count = dead.loc[dead['Race'] == 1].shape[0]
        values = [total, white_count, black_count]
        soft_finding = total == 177 and white_count == 121 and black_count == 56
        return (values, soft_finding, values)

    def finding_5_9(self):
        """Of the 177 that died, 33 were obese (18.64%) and 144 were not obese (81.36%) at baseline."""    
        dead = self.get_dead()
        total = dead.shape[0]
        obese_count = dead.loc[dead['Obesity'] == 1].shape[0]
        not_obese_count = dead.loc[dead['Obesity'] == 0].shape[0]
        values = [total, obese_count, not_obese_count]
        soft_finding = total == 177 and obese_count == 33 and not_obese_count == 144
        return (values, soft_finding, values)

    def finding_6_1(self):
        """In bivariate association, race was not associated with death due to cerebrovascular (unadjusted HR for Blacks compared to Whites = 0.78, 95% CI = 0.55-1.11), suggesting that Whites and Blacks had similar risk of future cerebrovascular mortality over 25 years."""
        corr_df = self.get_corr()
        corr_race_death = corr_df['Race'].loc['Death to cerebrovascular disease']
        soft_finding = abs(corr_race_death) < 0.05
        return ([corr_race_death], soft_finding, [corr_race_death])

    def finding_6_2(self):
        """In bivariate association, baseline obesity was not associated with future risk of cerebrovascular mortality (Unadjusted HR for Blacks compared to Whites = 0.84, 95% CI = 0.45-1.56), suggesting that Whites and Blacks had a similar risk of future cerebrovascular mortality over 25 years."""
        corr_df = self.get_corr()
        corr_obesity_death = corr_df['Obesity'].loc['Death to cerebrovascular disease']
        soft_finding = abs(corr_obesity_death) < 0.05
        return ([corr_obesity_death], soft_finding, [corr_obesity_death])

    # TODO: check that race correlation is for Black
    def finding_6_3(self):
        """Race (Black) was negatively associated with education and income"""
        corr_df = self.get_corr()
        values = [corr_df['Race'].loc['Education'], corr_df['Race'].loc['Income']]
        soft_finding = all(x < 0 for x in values)
        return (values, soft_finding, values)

    # TODO: check that race correlation is for Black
    def finding_6_4(self):
        """[race (Black) was]... positively associated with depressive symptoms, hypertension, and obesity."""
        corr_df = self.get_corr()
        values = [corr_df['Race'].loc['Depressive symptoms'], corr_df['Race'].loc['HTN'], corr_df['Race'].loc['Obesity']]
        soft_finding =  all(x > 0 for x in values)
        return (values, soft_finding, values)

    # TODO: check that race correlation is for Black
    def finding_6_5(self):
        """Blacks more frequently smoked and less frequently exercised.""" # implies positive correlation with smoking and negative with exercise
        corr_df = self.get_corr()
        values = [corr_df['Race'].loc['Smoking'], corr_df['Race'].loc['Exercise']]
        soft_finding = values[0] > 0 and  values[1] < 0
        return (values, soft_finding, values)

    # TODO: check that race correlation is for Black
    def finding_6_6(self):
        """Race was not associated with cerebrovascular death.""" # same as finding_6_1?
        corr_df = self.get_corr()
        corr_race_death = corr_df['Race'].loc['Death to cerebrovascular disease']
        soft_finding = abs(corr_race_death) < 0.05
        return ([corr_race_death], soft_finding, [corr_race_death])

    # TODO: check that gender correlation is for female
    def finding_6_7(self):
        """Baseline obesity was associated with female gender and less education, income, smoking, and exercise."""
        corr_df = self.get_corr()
        values = [corr_df['Obesity'].loc['Gender'], corr_df['Obesity'].loc['Education'], corr_df['Obesity'].loc['Income'], corr_df['Obesity'].loc['Smoking'], corr_df['Obesity'].loc['Exercise']]
        soft_finding =  values[0] > 0 and all(x < 0 for x in values[1:])
        return (values, soft_finding, values)

    def finding_6_8(self):
        """Obesity at baseline was associated with depressive symptoms and hypertension at baseline."""
        corr_df = self.get_corr()
        values = [corr_df['Obesity'].loc['Depressive symptoms'], corr_df['Obesity'].loc['HTN']]
        soft_finding = all(x > 0 for x in values)
        return (values, soft_finding, values)
    
    def finding_6_9(self):
        """Obesity at baseline was not associated with cerebrovascular death in the pooled sample (Table 2).""" # same as finding_6_2?
        corr_df = self.get_corr()
        corr_obesity_death = corr_df['Obesity'].loc['Death to cerebrovascular disease']
        soft_finding = abs(corr_obesity_death) < 0.05
        return ([corr_obesity_death], soft_finding, [corr_obesity_death])

    def finding_6_10(self):
        """According to Model 1 in the pooled sample, baseline obesity did not predict cerebrovascular mortality (HR = 0.86, 0.49-1.51), independent of demographic, socioeconomic, health behaviors, and health factors at baseline."""
        pass

    def finding_6_11(self):
        """According to Model 2, race interacted with baseline obesity on outcome (HR = 3.17, 1.09-9.21), suggesting a stronger association between baseline obesity and future risk for cerebrovascular deaths for Blacks, in comparison to Whites (Table 3)."""
        pass

    def finding_6_12(self):
        """As Model 3 shows, obesity did not predict the outcome in Whites (HR = 0.69, 0.31-1.53)."""
        pass

    def finding_6_13(self):
        """Model 4 shows that obesity predicts risk of cerebrovascular mortality for Blacks (HR = 2.51, 1.43-4.39) (Table 4)."""
        pass