import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.proportion import proportions_chisquare

from SynRD.publication import Publication, Finding, NonReproducibleFindingException, TAXONOMY


class Iverson22Football(Publication):
    """A class for the 'High School Football and Risk
    for Depression and Suicidality in Adulthood:
    Findings From a National Longitudinal Study'
    paper."""

    DEFAULT_PAPER_ATTRIBUTES = {
        'id': 'iverson22football',
        'length_pages': 9,
        'authors': ['Grant L. Iverson', 'Douglas P. Terry'],
        'journal': 'Frontiers in Neurology',
        'year': 2022,
        'current_citations': 0,
        'base_dataframe_pickle': 'iverson22football_dataframe.pickle'
    }

    DATAFRAME_COLUMNS = [
        "BIO_SEX", "S44A21", "H1GI9", "H5OD11", "S1", "IYEAR5", "IMONTH5", "H1GI1Y", "H1GI1M",
        "H5ID6G", "H5MN8", "S44A18", "S44A19", "S44A20", "S44A22", "S44A23", "S44A24",
        "S44A25", "S44A26", "S44A27", "S44A28", "S44A29", "H1HS3", "H1SU1", "H5ID6I", "H5ID13",
        "H5SS0B"
    ]

    INPUT_FILES = [
        'iverson22football/data/21600-0001-Data.tsv', 'iverson22football/data/21600-0032-Data.tsv'
    ]

    # FILENAME = "iverson22football"

    def __init__(self, dataframe=None):
        super(Iverson22Football, self).__init__(dataframe=dataframe)
        
        self.FINDINGS = self.FINDINGS + [
            Finding(self.finding_3_1, description="finding_3_1",
                    text="""The sample was, on average, 38 years old at the Wave V assessment.""",
                    finding_type=TAXONOMY.DESCRIPTIVE_STATISTICS),
            Finding(self.finding_3_2, description="finding_3_2",
                    text="""During the Wave V assessment, 307 (17.4%) men reported being
                            diagnosed with depression at some point in their life, 275 (15.6%) being
                            diagnosed with an anxiety disorder or panic disorder at some point in their
                            life, 211 (12.0%) having received psychological or emotional counseling
                            in the past 12 months, 125 (7.1%) reported seriously thinking about suicide
                            in the past year, and 101 (5.8%) reported feeling depressed in the previous week
                            (i.e., “a lot of the time” or “most of the time or all of the time” over the past 7 days).""",
                            finding_type=TAXONOMY.DESCRIPTIVE_STATISTICS),
            Finding(self.finding_3_3, description="finding_3_3",
                    text="""Examining responses the participants gave during the Wave I assessment when they were adolescents,
                            369 (20.9%) reported playing (or intending to play) football in high school and 952 (54.0%) reported not
                            intending to play football in high school. Of note, 441 participants (25% of the sample) did not answer
                            this question and were excluded from analyses pertaining to football participation.""",
                            finding_type=TAXONOMY.DESCRIPTIVE_STATISTICS),
            Finding(self.finding_4_1, description="finding_4_1",
                    text="""During Wave I, there were 174 boys (9.9%) who reported undergoing psychological counseling in the past year
                            while in high school. During the Wave V interview, ~24 years later, those individuals who underwent
                            psychological counseling during adolescence were much more likely to report (i) a lifetime history of
                            depression [37.4 vs. 15.3%, χ2 ((11)) = 53.17, p &lt; 0.001, OR= 3.31, 95% CI = 2.37-4.64],""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.TEMPORAL_FIXED_CLASS),
            Finding(self.finding_4_2, description="finding_4_2",
                    text="""(ii) a lifetime history of an anxiety disorder or panic disorder [31.0 vs. 14.9%, χ2 ((11)) = 41.12, p &lt; 0.001,
                            OR= 2.61, 95% CI = 1.94-3.50],""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.TEMPORAL_FIXED_CLASS),
            Finding(self.finding_4_3, description="finding_4_3",
                    text="""(iii) having received psychological counseling in the past 12 months [31.0 vs. 12.0%, χ2 ((11)) = 36.00, p &lt; 0.001,
                            OR= 2.70, 95% CI = 1.98-3.67],""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.TEMPORAL_FIXED_CLASS),
            Finding(self.finding_4_4, description="finding_4_4",
                    text="""(iv) suicidal ideation in the past year [12.6 vs. 6.4%, χ2(1) = 9.24 , 95%CI = 1.29–3.44],
                            , p = 0.002, OR = 2.11""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.TEMPORAL_FIXED_CLASS),
            # Finding(self.finding_4_5),
            Finding(self.finding_4_6, description="finding_4_6",
                    text="""When interviewed during adolescence, 186 boys (10.6%) endorsed thoughts of suicide in the past year.
                            At the followup assessment, ∼24 years later, those men who reported suicide ideation during adolescence,
                            compared to those who did not, weremore likely to report (i) a lifetime history of depression
                            [28.0 vs. 16.0%, χ2 (1) = 16.54, p &lt; 0.001, OR = 2.03, 95% CI = 1.44– 2.88]""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.TEMPORAL_FIXED_CLASS),
            Finding(self.finding_4_7, description="finding_4_7",
                    text="""(ii) having received psychological or emotional counseling in the past 12 months
                            [21.5 vs. 10.8%, χ2(1) = 17.95, p &lt; 0.001, OR= 2.26, 95% CI = 1.54–3.31]""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.TEMPORAL_FIXED_CLASS),
            Finding(self.finding_4_8, description="finding_4_8",
                    text="""(iii) suicide ideation in the past year
                            [19.4 vs. 5.6%, χ2(1) = 48.18, p &lt; 0.001, OR= 4.06, 95%CI = 2.66–6.20]""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.TEMPORAL_FIXED_CLASS),
            Finding(self.finding_4_9, description="finding_4_9",
                    text="""(iv) feeling depressed within the past 7 days
                            [13.0 vs. 4.8%, χ2 (1) = 20.79, p &lt; 0.001, OR = 2.97, 95% CI = 1.82– 4.84]""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.TEMPORAL_FIXED_CLASS),
            Finding(self.finding_4_10, description="finding_4_10",
                    text="""Those who reported suicidal ideation during adolescence had a higher lifetime history of anxiety
                            disorder or panic disorder, but this result was not statistically significant
                            [20.4 vs. 15.0%, χ2 (1) = 3.78, p = 0.052, OR = 1.46, 95% CI = 0.995–2.14]""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.TEMPORAL_FIXED_CLASS),
            Finding(self.finding_4_11, description="finding_4_11",
                    text="""Participants who played football, compared to participants who did not, had similar rates of
                            (i) being diagnosed with depression at some point in their life
                            [13.6 vs. 17.5%; χ2 (1) = 3.09, p = 0.08, OR = 0.74 95% CI = 0.52–1.04; see Figure 1]""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.BETWEEN_CLASS),
            Finding(self.finding_4_12, description="finding_4_12",
                    text="""(ii) being diagnosed with an anxiety disorder or panic disorder at some point in their life
                            [13.4 vs. 16.1%, χ2 (1) = 1.53, p = 0.22, OR= 0.80, 95%CI = 0.59–1.14]""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.BETWEEN_CLASS),
            Finding(self.finding_4_13, description="finding_4_13",
                    text="""(iii) having received psychological or emotional counseling in the past 12 months
                            [10.4 vs. 11.6%, χ2 (1) = 0.37, p = 0.54, OR= 0.89, 95% CI = 0.60–1.31]""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.BETWEEN_CLASS),
            Finding(self.finding_4_14, description="finding_4_14",
                    text="""(iv) suicidal ideation in the past year
                            [6.0 vs. 7.0%; χ2 (1) = 0.49, p = 0.48, OR = 0.84, 95% CI = 0.51–1.38]""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.BETWEEN_CLASS),
            Finding(self.finding_4_15, description="finding_4_15",
                    text="""(v) feeling depressed in the past 7 days
                            [4.1 vs. 6.2%; χ2 2.12, p = 0.15, OR = 0.65, 95% CI = 0.37–1.17]. (1)""",
                            finding_type=TAXONOMY.MEAN_DIFFERENCE.value.BETWEEN_CLASS),
        ]

    def _recreate_dataframe(self, filename='iverson22football_dataframe.pickle'):
        wave1 = pd.read_csv(self.INPUT_FILES[0], sep='\t', skipinitialspace=True)
        wave5 = pd.read_csv(self.INPUT_FILES[1], sep='\t', skipinitialspace=True)

        df = wave1.merge(wave5, how='inner', on="AID")
        df = df[self.DATAFRAME_COLUMNS]
        df = df[df["BIO_SEX"] == 1]
        df = df[df["H5ID6G"].isin([0, 1])]
        df = df[df["H5MN8"].isin([0, 1])]

        df.to_pickle(filename)

        return df

    def finding_3_1(self):
        """
        The sample was, on average, 38 years old at the Wave V assessment.
        """
        average_age_wave_5 = ((self.dataframe["IYEAR5"] + 1 / self.dataframe["IMONTH5"]) - (
                self.dataframe["H1GI1Y"] + 1 / self.dataframe["H1GI1M"])).mean() - 1900

        findings = [average_age_wave_5]
        soft_findings = [37 < average_age_wave_5 < 39]
        hard_findings = [37.9 < average_age_wave_5 < 38.1]
        return findings, soft_findings, hard_findings

    def finding_3_2(self):
        """
        During the Wave V assessment, 307 (17.4%) men reported being
        diagnosed with depression at some point in their life, 275 (15.6%) being
        diagnosed with an anxiety disorder or panic disorder at some point in their
        life, 211 (12.0%) having received psychological or emotional counseling
        in the past 12 months, 125 (7.1%) reported seriously thinking about suicide
        in the past year, and 101 (5.8%) reported feeling depressed in the previous week
        (i.e., “a lot of the time” or “most of the time or all of the time” over the past 7 days).
        """

        men_with_depression = len(self.dataframe[self.dataframe["H5ID6G"] == 1])
        men_with_depression_proportion = np.round((men_with_depression / len(self.dataframe)) * 100, 1)

        men_with_anxiety = len(self.dataframe[self.dataframe["H5ID6I"] == 1])
        men_with_anxiety_proportion = np.round((men_with_anxiety / len(self.dataframe)) * 100, 1)

        men_with_psycho_counseling = len(self.dataframe[self.dataframe["H5ID13"] == 1])
        men_with_psycho_counseling_proportion = np.round((men_with_psycho_counseling / len(self.dataframe)) * 100, 1)

        men_with_suicide = len(self.dataframe[self.dataframe["H5MN8"] == 1])
        men_with_suicide_proportion = np.round((men_with_suicide / len(self.dataframe)) * 100, 1)

        men_with_depression_past_week = len(self.dataframe[self.dataframe["H5SS0B"].isin([3, 4])])
        men_with_depression_past_week_proportion = np.round((men_with_depression_past_week / len(self.dataframe)) * 100,
                                                            1)

        findings = [(men_with_depression, men_with_depression_proportion),
                    (men_with_anxiety, men_with_anxiety_proportion),
                    (men_with_psycho_counseling, men_with_psycho_counseling_proportion), (men_with_suicide,
                                                                                          men_with_suicide_proportion),
                    (men_with_depression_past_week,
                     men_with_depression_past_week_proportion)]

        soft_findings = []
        hard_findings = [men_with_depression == 307, men_with_anxiety == 275, men_with_psycho_counseling == 211,
                         men_with_suicide == 125, men_with_depression_past_week == 101]
        return findings, soft_findings, hard_findings

    def finding_3_3(self):
        """
        Examining responses the participants gave during the Wave I assessment when they were adolescents,
        369 (20.9%) reported playing (or intending to play) football in high school and 952 (54.0%) reported not
        intending to play football in high school. Of note, 441 participants (25% of the sample) did not answer
        this question and were excluded from analyses pertaining to football participation.
        """
        self.played_football = len(self.dataframe[self.dataframe['S44A21'] == 1])
        played_football_proportion = np.round((self.played_football / len(self.dataframe)) * 100, 1)

        self.did_not_play_football = len(self.dataframe[self.dataframe['S44A21'] == 0])
        did_not_play_football_proportion = np.round((self.did_not_play_football / len(self.dataframe)) * 100, 1)

        did_not_answer = len(self.dataframe[(self.dataframe['S44A21'] != 0) & (self.dataframe['S44A21'] != 1)])
        did_not_answer_proportion = np.round((did_not_answer / len(self.dataframe)) * 100, 1)

        findings = [(self.played_football, played_football_proportion),
                    (self.did_not_play_football, did_not_play_football_proportion),
                    (did_not_answer, did_not_answer_proportion)]

        soft_findings = []
        hard_findings = [self.played_football == 369, self.did_not_play_football == 952, did_not_answer == 441]
        return findings, soft_findings, hard_findings

    def _calculate_statistics_for_psycho_counseling(self, column):
        psycho_counseling_proportion = np.round((len(
            self.psycho_counseling[self.psycho_counseling[column] == 1]) / len(self.psycho_counseling)) * 100, 1)

        normal_proportion = np.round((len(
            self.no_psycho_counseling[self.no_psycho_counseling[column] == 1]) / len(
            self.no_psycho_counseling)) * 100, 1)

        chi_square, p_value, (contigency_table, _) = proportions_chisquare(count=[len(
            self.psycho_counseling[self.psycho_counseling[column] == 1]),
            len(self.no_psycho_counseling[self.no_psycho_counseling[column] == 1])],
            nobs=[len(self.psycho_counseling), len(self.no_psycho_counseling)])

        odds_ratio, _ = stats.fisher_exact(contigency_table)
        confidence_interval = Table2x2(contigency_table).oddsratio_confint()

        return psycho_counseling_proportion, normal_proportion, \
               chi_square, p_value, odds_ratio, confidence_interval

    def finding_4_1(self):
        """
        During Wave I, there were 174 boys (9.9%) who reported undergoing psychological counseling in the past year
        while in high school. During the Wave V interview, ∼24 years later, those individuals who underwent
        psychological counseling during adolescence were much more likely to report (i) a lifetime history of
        depression [37.4 vs. 15.3%, χ2 ((11)) = 53.17, p &lt; 0.001, OR= 3.31, 95% CI = 2.37–4.64],
        """
        self.psycho_counseling = self.dataframe[self.dataframe["H1HS3"] == 1]
        self.no_psycho_counseling = self.dataframe[self.dataframe["H1HS3"] == 0]

        boys_psycho_counseling_past_year = len(self.psycho_counseling)
        boys_psycho_counseling_past_year_proportion = np.round(
            (boys_psycho_counseling_past_year / len(self.dataframe)) * 100, 1)

        psycho_counseling_depression_proportion, normal_depression_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_psycho_counseling(column="H5ID6G")

        findings = [(boys_psycho_counseling_past_year, boys_psycho_counseling_past_year_proportion),
                    psycho_counseling_depression_proportion, normal_depression_proportion]
        soft_findings = [psycho_counseling_depression_proportion > normal_depression_proportion]
        hard_findings = [boys_psycho_counseling_past_year == 174,
                         np.allclose(boys_psycho_counseling_past_year_proportion, 9.9, atol=10e-2),
                         np.allclose(psycho_counseling_depression_proportion, 37.4, atol=10e-2),
                         np.allclose(normal_depression_proportion, 15.3, atol=10e-2),
                         np.allclose(chi_square, 53.17, atol=0.1),
                         p_value < 0.001,
                         np.allclose(odds_ratio, 3.31, atol=0.1),
                         np.allclose(confidence_interval[0], 2.37, atol=0.1),
                         np.allclose(confidence_interval[1], 4.64, atol=0.1)]

        return findings, soft_findings, hard_findings

    def finding_4_2(self):
        """
        (1) , p & lt; 0.001, (ii) a lifetime history of anxiety disorder or panic disorder[27.0 vs.
        14.4 %, χ2= 18.88 OR =  2.20, 95 % CI = 1.53–3.16],
        """
        psycho_counseling_anxiety_proportion, normal_anxiety_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_psycho_counseling(column="H5ID6I")

        findings = [psycho_counseling_anxiety_proportion, normal_anxiety_proportion]
        soft_findings = [psycho_counseling_anxiety_proportion > normal_anxiety_proportion]
        hard_findings = [np.allclose(psycho_counseling_anxiety_proportion, 27.0, atol=10e-2),
                         np.allclose(normal_anxiety_proportion, 14.4, atol=10e-2),
                         np.allclose(chi_square, 18.88, atol=0.12),
                         p_value < 0.001,
                         np.allclose(odds_ratio, 2.20, atol=0.1),
                         np.allclose(confidence_interval[0], 1.53, atol=0.1),
                         np.allclose(confidence_interval[1], 3.16, atol=0.1)]

        return findings, soft_findings, hard_findings

    def finding_4_3(self):
        """
        (1) (iii) having received psychological or emotional counseling in the past 12 months
        [21.3 vs. 11.0%, χ2 = 14.48, p &lt; 0.001, OR= 2.18, 95%CI = 1.47–3.24],
        """

        psycho_counseling_emotional_proportion, normal_emotional_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_psycho_counseling(column="H5ID13")

        findings = [psycho_counseling_emotional_proportion, normal_emotional_proportion]
        soft_findings = [psycho_counseling_emotional_proportion > normal_emotional_proportion]
        hard_findings = [np.allclose(psycho_counseling_emotional_proportion, 21.3, atol=10e-2),
                         np.allclose(normal_emotional_proportion, 11.0, atol=10e-2),
                         np.allclose(chi_square, 14.48, atol=0.1),  # here real chi square = 15,75
                         p_value < 0.001,
                         np.allclose(odds_ratio, 2.18, atol=0.1),
                         np.allclose(confidence_interval[0], 1.47, atol=0.1),
                         np.allclose(confidence_interval[1], 3.24, atol=0.1)]

        return findings, soft_findings, hard_findings

    def finding_4_4(self):
        """
        (iv) suicidal ideation in the past year [12.6 vs. 6.4%, χ2(1) = 9.24 , 95%CI = 1.29–3.44],
         , p = 0.002, OR = 2.11
        """
        psycho_counseling_suicidal_proportion, normal_suicidal_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_psycho_counseling(column="H5MN8")

        findings = [psycho_counseling_suicidal_proportion, normal_suicidal_proportion]
        soft_findings = [psycho_counseling_suicidal_proportion > normal_suicidal_proportion]
        hard_findings = [np.allclose(psycho_counseling_suicidal_proportion, 12.6, atol=10e-2),
                         np.allclose(normal_suicidal_proportion, 6.4, atol=10e-2),
                         np.allclose(chi_square, 9.24, atol=0.1),
                         p_value < 0.0025,
                         np.allclose(odds_ratio, 2.11, atol=0.1),
                         np.allclose(confidence_interval[0], 1.29, atol=0.1),
                         np.allclose(confidence_interval[1], 3.44, atol=0.1)]

        return findings, soft_findings, hard_findings

    def finding_4_5(self):
        """
        (v) current depression [11.6 vs. 5.1%, χ2 (1) = 11.77, p &lt; 0.001, OR = 2.41, 95% CI = 1.43–4.04],
        compared to those who did not
        """
        raise NonReproducibleFindingException

    def _calculate_statistics_for_suicide(self, column):
        suicide_proportion = np.round((len(
            self.suicide[self.suicide[column] == 1]) / len(self.suicide)) * 100, 1)
        normal_proportion = np.round((len(
            self.no_suicide[self.no_suicide[column] == 1]) / len(
            self.no_suicide)) * 100, 1)

        chi_square, p_value, (contigency_table, _) = proportions_chisquare(count=[len(
            self.suicide[self.suicide[column] == 1]),
            len(self.no_suicide[self.no_suicide[column] == 1])],
            nobs=[len(self.suicide), len(self.no_suicide)])

        odds_ratio, _ = stats.fisher_exact(contigency_table)
        confidence_interval = Table2x2(contigency_table).oddsratio_confint()

        return suicide_proportion, normal_proportion, \
               chi_square, p_value, odds_ratio, confidence_interval

    def finding_4_6(self):
        """
        When interviewed during adolescence, 186 boys (10.6%) endorsed thoughts of suicide in the past year.
        At the followup assessment, ∼24 years later, those men who reported suicide ideation during adolescence,
        compared to those who did not, weremore likely to report (i) a lifetime history of depression
        [28.0 vs. 16.0%, χ2 (1) = 16.54, p &lt; 0.001, OR = 2.03, 95% CI = 1.44– 2.88]
        """
        self.suicide = self.dataframe[self.dataframe["H1SU1"] == 1]
        self.no_suicide = self.dataframe[self.dataframe["H1SU1"] == 0]

        boys_suicide_past_year = len(self.psycho_counseling)
        boys_suicide_past_year_proportion = np.round((boys_suicide_past_year / len(self.dataframe)) * 100, 1)

        suicide_depression_proportion, normal_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_suicide(column="H5ID6G")

        findings = [(boys_suicide_past_year, boys_suicide_past_year_proportion),
                    suicide_depression_proportion, normal_proportion]
        soft_findings = [suicide_depression_proportion > normal_proportion]
        hard_findings = [boys_suicide_past_year == 186,
                         np.allclose(boys_suicide_past_year_proportion, 10.6, atol=10e-2),
                         np.allclose(suicide_depression_proportion, 28.0, atol=10e-2),
                         np.allclose(normal_proportion, 16.0, atol=10e-2),
                         np.allclose(chi_square, 16.54, atol=0.1),
                         p_value < 0.001,
                         np.allclose(odds_ratio, 2.03, atol=0.1),
                         np.allclose(confidence_interval[0], 1.44, atol=0.1),
                         np.allclose(confidence_interval[1], 2.88, atol=0.1)]

        return findings, soft_findings, hard_findings

    def finding_4_7(self):
        """
        (ii) having received psychological or emotional counseling in the past 12 months
        [21.5 vs. 10.8%, χ2(1) = 17.95, p &lt; 0.001, OR= 2.26, 95% CI = 1.54–3.31]
        """
        suicide_emotional_proportion, normal_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_suicide(column="H5ID13")

        findings = [suicide_emotional_proportion, normal_proportion]
        soft_findings = [suicide_emotional_proportion > normal_proportion]
        hard_findings = [np.allclose(suicide_emotional_proportion, 21.5, atol=10e-2),
                         np.allclose(normal_proportion, 10.8, atol=10e-2),
                         np.allclose(chi_square, 17.95, atol=0.1),  # 18.25
                         p_value < 0.001,
                         np.allclose(odds_ratio, 2.26, atol=0.1),
                         np.allclose(confidence_interval[0], 1.54, atol=0.1),
                         np.allclose(confidence_interval[1], 3.31, atol=0.1)]

        return findings, soft_findings, hard_findings

    def finding_4_8(self):
        """
        (iii) suicide ideation in the past year
        [19.4 vs. 5.6%, χ2(1) = 48.18, p &lt; 0.001, OR= 4.06, 95%CI = 2.66–6.20]
        """
        suicide_suicide_proportion, normal_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_suicide(column="H5MN8")

        findings = [suicide_suicide_proportion, normal_proportion]
        soft_findings = [suicide_suicide_proportion > normal_proportion]
        hard_findings = [np.allclose(suicide_suicide_proportion, 19.4, atol=10e-2),
                         np.allclose(normal_proportion, 5.6, atol=10e-2),
                         np.allclose(chi_square, 48.18, atol=0.1),
                         p_value < 0.001,
                         np.allclose(odds_ratio, 4.06, atol=0.1),
                         np.allclose(confidence_interval[0], 2.66, atol=0.1),
                         np.allclose(confidence_interval[1], 6.20, atol=0.1)]

        return findings, soft_findings, hard_findings

    # TODO: check this: ([61.8, 73.5], [False], [False, False, False, True, False, False, False])
    def finding_4_9(self):
        """
        (iv) feeling depressed within the past 7 days
        [13.0 vs. 4.8%, χ2 (1) = 20.79, p &lt; 0.001, OR = 2.97, 95% CI = 1.82– 4.84]
        """
        suicide_depressed_proportion, normal_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_suicide(column="H5SS0B")

        findings = [suicide_depressed_proportion, normal_proportion]
        soft_findings = [suicide_depressed_proportion > normal_proportion]
        hard_findings = [np.allclose(suicide_depressed_proportion, 13.0, atol=10e-2),
                         np.allclose(normal_proportion, 4.8, atol=10e-2),
                         np.allclose(chi_square, 20.79, atol=0.1),
                         p_value < 0.001,
                         np.allclose(odds_ratio, 2.97, atol=0.1),
                         np.allclose(confidence_interval[0], 1.82, atol=0.1),
                         np.allclose(confidence_interval[1], 4.84, atol=0.1)]

        return findings, soft_findings, hard_findings

    def finding_4_10(self):
        """
        Those who reported suicidal ideation during adolescence had a higher lifetime history of anxiety
        disorder or panic disorder, but this result was not statistically significant
        [20.4 vs. 15.0%, χ2 (1) = 3.78, p = 0.052, OR = 1.46, 95% CI = 0.995–2.14]
        """
        suicide_anxiety_proportion, normal_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_suicide(column="H5ID6I")

        findings = [suicide_anxiety_proportion, normal_proportion]
        soft_findings = [suicide_anxiety_proportion > normal_proportion]
        hard_findings = [np.allclose(suicide_anxiety_proportion, 20.4, atol=10e-2),
                         np.allclose(normal_proportion, 15.0, atol=10e-2),
                         np.allclose(chi_square, 3.78, atol=0.1),
                         p_value < 0.06,
                         np.allclose(odds_ratio, 1.46, atol=0.1),
                         np.allclose(confidence_interval[0], 0.995, atol=0.1),
                         np.allclose(confidence_interval[1], 2.14, atol=0.1)]

        return findings, soft_findings, hard_findings

    def _calculate_statistics_for_football(self, column):
        football_proportion = np.round((len(
            self.played_football[self.played_football[column] == 1]) / len(self.played_football)) * 100, 1)
        no_football_proportion = np.round((len(
            self.did_not_play_football[self.did_not_play_football[column] == 1]) / len(
            self.did_not_play_football)) * 100, 1)

        chi_square, p_value, (contigency_table, _) = proportions_chisquare(count=[len(
            self.played_football[self.played_football[column] == 1]),
            len(self.did_not_play_football[self.did_not_play_football[column] == 1])],
            nobs=[len(self.played_football), len(self.did_not_play_football)])

        odds_ratio, _ = stats.fisher_exact(contigency_table)
        confidence_interval = Table2x2(contigency_table).oddsratio_confint()

        return football_proportion, no_football_proportion, \
               chi_square, p_value, odds_ratio, confidence_interval

    # TODO: check those findings 4_11-4_15

    def finding_4_11(self):
        """
        Participants who played football, compared to participants who did not, had similar rates of
        (i) being diagnosed with depression at some point in their life
        [13.6 vs. 17.5%; χ2 (1) = 3.09, p = 0.08, OR = 0.74 95% CI = 0.52–1.04; see Figure 1]
        """
        football_depression_proportion, no_football_depression_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_suicide(column="H5ID6G")

        findings = [football_depression_proportion, no_football_depression_proportion]
        soft_findings = [football_depression_proportion < no_football_depression_proportion]
        hard_findings = [np.allclose(football_depression_proportion, 13.6, atol=10e-2),
                         np.allclose(no_football_depression_proportion, 17.5, atol=10e-2),
                         np.allclose(chi_square, 3.09, atol=0.1),
                         p_value < 0.09,
                         np.allclose(odds_ratio, 0.74, atol=0.1),
                         np.allclose(confidence_interval[0], 0.52, atol=0.1),
                         np.allclose(confidence_interval[1], 1.04, atol=0.1)]

        return findings, soft_findings, hard_findings

    def finding_4_12(self):
        """
        (ii) being diagnosed with an anxiety disorder or panic disorder at some point in their life
        [13.4 vs. 16.1%, χ2 (1) = 1.53, p = 0.22, OR= 0.80, 95%CI = 0.59–1.14]
        """
        football_anxiety_proportion, no_football_anxiety_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_suicide(column="H5ID6I")

        findings = [football_anxiety_proportion, no_football_anxiety_proportion]
        soft_findings = [football_anxiety_proportion < no_football_anxiety_proportion]
        hard_findings = [np.allclose(football_anxiety_proportion, 13.4, atol=10e-2),
                         np.allclose(no_football_anxiety_proportion, 16.1, atol=10e-2),
                         np.allclose(chi_square, 3.09, atol=0.1),
                         p_value < 0.3,
                         np.allclose(odds_ratio, 0.80, atol=0.1),
                         np.allclose(confidence_interval[0], 0.59, atol=0.1),
                         np.allclose(confidence_interval[1], 1.14, atol=0.1)]

        return findings, soft_findings, hard_findings

    def finding_4_13(self):
        """
        (iii) having received psychological or emotional counseling in the past 12 months
        [10.4 vs. 11.6%, χ2 (1) = 0.37, p = 0.54, OR= 0.89, 95% CI = 0.60–1.31]
        """
        football_emotional_proportion, no_football_emotional_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_suicide(column="H5ID13")

        findings = [football_emotional_proportion, no_football_emotional_proportion]
        soft_findings = [football_emotional_proportion < no_football_emotional_proportion]
        hard_findings = [np.allclose(football_emotional_proportion, 10.4, atol=10e-2),
                         np.allclose(no_football_emotional_proportion, 11.6, atol=10e-2),
                         np.allclose(chi_square, 3.09, atol=0.1),
                         p_value < 0.4,
                         np.allclose(odds_ratio, 0.89, atol=0.1),
                         np.allclose(confidence_interval[0], 0.60, atol=0.1),
                         np.allclose(confidence_interval[1], 1.31, atol=0.1)]

        return findings, soft_findings, hard_findings

    def finding_4_14(self):
        """
        (iv) suicidal ideation in the past year
        [6.0 vs. 7.0%; χ2 (1) = 0.49, p = 0.48, OR = 0.84, 95% CI = 0.51–1.38]
        """
        football_suicidal_proportion, no_football_suicidal_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_suicide(column="H5MN8")

        findings = [football_suicidal_proportion, no_football_suicidal_proportion]
        soft_findings = [football_suicidal_proportion < no_football_suicidal_proportion]
        hard_findings = [np.allclose(football_suicidal_proportion, 6.0, atol=10e-2),
                         np.allclose(no_football_suicidal_proportion, 7.0, atol=10e-2),
                         np.allclose(chi_square, 0.49, atol=0.1),
                         p_value < 0.5,
                         np.allclose(odds_ratio, 0.84, atol=0.1),
                         np.allclose(confidence_interval[0], 0.51, atol=0.1),
                         np.allclose(confidence_interval[1], 1.38, atol=0.1)]

        return findings, soft_findings, hard_findings

    def finding_4_15(self):
        """
        (v) feeling depressed in the past 7 days
        [4.1 vs. 6.2%; χ2 2.12, p = 0.15, OR = 0.65, 95% CI = 0.37–1.17]. (1)
        """
        football_depressed_proportion, no_football_depressed_proportion, chi_square, \
        p_value, odds_ratio, confidence_interval = self._calculate_statistics_for_suicide(column="H5SS0B")

        findings = [football_depressed_proportion, no_football_depressed_proportion]
        soft_findings = [football_depressed_proportion < no_football_depressed_proportion]
        hard_findings = [np.allclose(football_depressed_proportion, 4.1, atol=10e-2),
                         np.allclose(no_football_depressed_proportion, 6.2, atol=10e-2),
                         np.allclose(chi_square, 2.12, atol=0.1),
                         p_value < 0.16,
                         np.allclose(odds_ratio, 0.65, atol=0.1),
                         np.allclose(confidence_interval[0], 0.37, atol=0.1),
                         np.allclose(confidence_interval[1], 1.17, atol=0.1)]

        return findings, soft_findings, hard_findings

    # TODO: wrap this up to the VisualFinding
    def figure_finding(self):
        all_depression = round(len(self.dataframe[self.dataframe["H5ID6G"] == 1]) / len(self.dataframe) * 100, 1)
        all_suicide = round(len(self.dataframe[self.dataframe["H5MN8"] == 1]) / len(self.dataframe) * 100, 1)

        played_football = self.dataframe[self.dataframe['S44A21'] == 1]
        football_depression = round(len(played_football[played_football["H5ID6G"] == 1]) / len(played_football) * 100,
                                    1)
        football_suicide = round(len(played_football[played_football["H5MN8"] == 1]) / len(played_football) * 100, 1)

        did_not_play_football = self.dataframe[self.dataframe['S44A21'] == 0]
        no_football_depression = round(
            len(did_not_play_football[did_not_play_football["H5ID6G"] == 1]) / len(did_not_play_football) * 100, 1)
        no_football_suicide = round(
            len(did_not_play_football[did_not_play_football["H5MN8"] == 1]) / len(did_not_play_football) * 100, 1)

        no_sports = self.dataframe[
            (self.dataframe["S44A18"] == 0) & (self.dataframe["S44A19"] == 0) & (self.dataframe["S44A20"] == 0) & (
                        self.dataframe["S44A21"] == 0) & (
                    self.dataframe["S44A22"] == 0) & (self.dataframe["S44A23"] == 0) & (
                        self.dataframe["S44A24"] == 0) & (self.dataframe["S44A25"] == 0) & (
                    self.dataframe["S44A26"] == 0) & (self.dataframe["S44A27"] == 0) & (
                        self.dataframe["S44A28"] == 0) & (self.dataframe["S44A29"] == 0)]

        no_sports_depression = round(len(no_sports[no_sports["H5ID6G"] == 1]) / len(no_sports) * 100, 1)
        no_sports_suicide = round(len(no_sports[no_sports["H5MN8"] == 1]) / len(no_sports) * 100, 1)

        psycho_help = self.dataframe[self.dataframe["H1HS3"] == 1]
        psycho_help_depression = round(len(psycho_help[psycho_help["H5ID6G"] == 1]) / len(psycho_help) * 100, 1)
        psycho_help_suicide = round(len(psycho_help[psycho_help["H5MN8"] == 1]) / len(psycho_help) * 100, 1)

        suicide_adolscence = self.dataframe[self.dataframe["H1SU1"] == 1]
        suicide_adolscence_depression = round(
            len(suicide_adolscence[suicide_adolscence["H5ID6G"] == 1]) / len(suicide_adolscence) * 100, 1)
        suicide_adolscence_suicide = round(
            len(suicide_adolscence[suicide_adolscence["H5MN8"] == 1]) / len(suicide_adolscence) * 100, 1)

        labels = ["Total Sample", "High School Football", "No High School Football", "No Sports in High School",
                  "Psychological Treatment During Adolescence", "Suicide Ideation During Adolescence"]

        list(range(6))
        values = [all_depression, all_suicide, football_depression, football_suicide,
                  no_football_depression, no_football_suicide, no_sports_depression, no_sports_suicide,
                  psycho_help_depression, psycho_help_suicide, suicide_adolscence_depression,
                  suicide_adolscence_suicide]

        depression_values = values[::2]
        suicide_values = values[1::2]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, depression_values, width, label='Lifetime History of Depression')
        rects2 = ax.bar(x + width / 2, suicide_values, width, label='Suicide Ideation Past Year')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title('Scores by group and gender')
        # plt.xticks(ticks = tickvalues ,labels = labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()
        plt.show()

    # TODO: wrap this up to VisualFinding or some kind of TableFinding
    def table_finding(self):
        white = len(self.dataframe[self.dataframe["H1GI9"] == 1])
        black_african_american = len(self.dataframe[self.dataframe["H1GI9"] == 2])
        indian_native = len(self.dataframe[self.dataframe["H1GI9"] == 4])
        asian_pacific = len(self.dataframe[self.dataframe["H1GI9"] == 3])
        other = len(self.dataframe[self.dataframe["H1GI9"] == 5])
        dont_know = len(self.dataframe[self.dataframe["H1GI9"] == 8])

        total = len(self.dataframe)

        print("========================= Race =========================")
        print("========= All =========")
        print(f"White: {white} ({round(white / total * 100, 1)}%)")
        print(f"Black: {black_african_american} ({round(black_african_american / total * 100, 1)}%)")
        print(f"Indian: {indian_native} ({round(indian_native / total * 100, 1)}%)")
        print(f"Asian: {asian_pacific} ({round(asian_pacific / total * 100, 1)}%)")
        print(f"Other: {other} ({round(other / total * 100, 1)}%)")
        print(f"Don't know: {dont_know} ({round(dont_know / total * 100, 1)}%)")

        print("========= Played football =========")
        football = self.dataframe[self.dataframe['S44A21'] == 1]
        white = len(football[football["H1GI9"] == 1])
        black_african_american = len(football[football["H1GI9"] == 2])
        indian_native = len(football[football["H1GI9"] == 4])
        asian_pacific = len(football[football["H1GI9"] == 3])
        other = len(football[football["H1GI9"] == 5])
        dont_know = len(football[football["H1GI9"] == 8])

        total = len(football)

        print(f"White: {white} ({round(white / total * 100, 1)}%)")
        print(f"Black: {black_african_american} ({round(black_african_american / total * 100, 1)}%)")
        print(f"Indian: {indian_native} ({round(indian_native / total * 100, 1)}%)")
        print(f"Asian: {asian_pacific} ({round(asian_pacific / total * 100, 1)}%)")
        print(f"Other: {other} ({round(other / total * 100, 1)}%)")
        print(f"Don't know: {dont_know} ({round(dont_know / total * 100, 1)}%)")

        print("========= Didn't play football =========")
        no_football = self.dataframe[self.dataframe['S44A21'] == 0]
        white = len(no_football[no_football["H1GI9"] == 1])
        black_african_american = len(no_football[no_football["H1GI9"] == 2])
        indian_native = len(no_football[no_football["H1GI9"] == 4])
        asian_pacific = len(no_football[no_football["H1GI9"] == 3])
        other = len(no_football[no_football["H1GI9"] == 5])
        dont_know = len(no_football[no_football["H1GI9"] == 8])

        total = len(no_football)

        print(f"White: {white} ({round(white / total * 100, 1)}%)")
        print(f"Black: {black_african_american} ({round(black_african_american / total * 100, 1)}%)")
        print(f"Indian: {indian_native} ({round(indian_native / total * 100, 1)}%)")
        print(f"Asian: {asian_pacific} ({round(asian_pacific / total * 100, 1)}%)")
        print(f"Other: {other} ({round(other / total * 100, 1)}%)")
        print(f"Don't know: {dont_know} ({round(dont_know / total * 100, 1)}%)")

        print("========================= Education =========================")
        print("========= All =========")
        some_high = len(self.dataframe[self.dataframe["H5OD11"] == 2])
        high_diploma = len(self.dataframe[self.dataframe["H5OD11"].isin([3, 4])])
        college = len(self.dataframe[self.dataframe["H5OD11"].isin([5, 6, 7, 8, 9])])
        bachelor = len(self.dataframe[self.dataframe["H5OD11"] == 10])
        more_bachelor = len(self.dataframe[self.dataframe["H5OD11"].isin([11, 12, 13, 14, 15, 16])])

        total = some_high + high_diploma + college + bachelor + more_bachelor

        print(f"Some high school: {some_high} ({round(some_high / total * 100, 1)}%)")
        print(
            f"High school diploma/general equivalency diploma: {high_diploma} ({round(high_diploma / total * 100, 1)}%)")
        print(f"Some college/vocational school/associate’s degree: {college} ({round(college / total * 100, 1)}%)")
        print(f"Bachelor’s degree: {bachelor} ({round(bachelor / total * 100, 1)}%)")
        print(f"More than bachelor’s degree: {more_bachelor} ({round(more_bachelor / total * 100, 1)}%)")

        print("========= Played football =========")
        some_high = len(football[football["H5OD11"] == 2])
        high_diploma = len(football[football["H5OD11"].isin([3, 4])])
        college = len(football[football["H5OD11"].isin([5, 6, 7, 8, 9])])
        bachelor = len(football[football["H5OD11"] == 10])
        more_bachelor = len(football[football["H5OD11"].isin([11, 12, 13, 14, 15, 16])])

        total = some_high + high_diploma + college + bachelor + more_bachelor

        print(f"Some high school: {some_high} ({round(some_high / total * 100, 1)}%)")
        print(
            f"High school diploma/general equivalency diploma: {high_diploma} ({round(high_diploma / total * 100, 1)}%)")
        print(f"Some college/vocational school/associate’s degree: {college} ({round(college / total * 100, 1)}%)")
        print(f"Bachelor’s degree: {bachelor} ({round(bachelor / total * 100, 1)}%)")
        print(f"More than bachelor’s degree: {more_bachelor} ({round(more_bachelor / total * 100, 1)}%)")

        print("========= Didn't play football =========")

        some_high = len(no_football[no_football["H5OD11"] == 2])
        high_diploma = len(no_football[no_football["H5OD11"].isin([3, 4])])
        college = len(no_football[no_football["H5OD11"].isin([5, 6, 7, 8, 9])])
        bachelor = len(no_football[no_football["H5OD11"] == 10])
        more_bachelor = len(no_football[no_football["H5OD11"].isin([11, 12, 13, 14, 15, 16])])

        total = some_high + high_diploma + college + bachelor + more_bachelor

        print(f"Some high school: {some_high} ({round(some_high / total * 100, 1)}%)")
        print(
            f"High school diploma/general equivalency diploma: {high_diploma} ({round(high_diploma / total * 100, 1)}%)")
        print(f"Some college/vocational school/associate’s degree: {college} ({round(college / total * 100, 1)}%)")
        print(f"Bachelor’s degree: {bachelor} ({round(bachelor / total * 100, 1)}%)")
        print(f"More than bachelor’s degree: {more_bachelor} ({round(more_bachelor / total * 100, 1)}%)")

        print("========================= Age =========================")
        print("========= Wave 1 =========")
        # means
        all_age_wave1 = round(self.dataframe["S1"].dropna().mean(), 2)
        football_age_wave1 = round(football["S1"].dropna().mean(), 2)
        no_football_age_wave1 = round(no_football["S1"].dropna().mean(), 2)

        # stds
        std_all_age_wave1 = round(self.dataframe["S1"].dropna().std(), 2)
        std_football_age_wave1 = round(football["S1"].dropna().std(), 2)
        std_no_football_age_wave1 = round(no_football["S1"].dropna().std(), 2)

        # range
        range1_all_min = self.dataframe["S1"].min()
        range1_all_max = self.dataframe["S1"].max()
        range1_football_min = football["S1"].min()
        range1_football_max = football["S1"].max()
        range1_no_football_min = no_football["S1"].min()
        range1_no_football_max = no_football["S1"].max()

        # interquartile range
        range_q_all_min, range_q_all_max = np.percentile(self.dataframe["S1"].dropna(), [25, 75])
        range_q_football_min, range_q_football_max = np.percentile(football["S1"].dropna(), [25, 75])
        range_q_no_football_min, range_q_no_football_max = np.percentile(no_football["S1"].dropna(), [25, 75])

        # medians
        med_all_age_wave1 = self.dataframe["S1"].median()
        med_football_age_wave1 = football["S1"].median()
        med_no_football_age_wave1 = no_football["S1"].median()

        print("================================= Age Wave1  =================================")
        print(f"Mean               : {all_age_wave1} | {football_age_wave1} | {no_football_age_wave1}")
        print(f"Std                : {std_all_age_wave1} | {std_football_age_wave1} | {std_no_football_age_wave1}")
        print(f"Median             : {med_all_age_wave1} | {med_football_age_wave1} | {med_no_football_age_wave1}")
        print(
            f"Interquartile Range: {range_q_all_min}-{range_q_all_max} | {range_q_football_min}-{range_q_football_max} | {range_q_no_football_min}-{range_q_no_football_max}")
        print(
            f"Range              : {range1_all_min}-{range1_all_max} | {range1_football_min}-{range1_football_max} | {range1_no_football_min}-{range1_no_football_max}")

        print("========= Wave 5 =========")

        # means
        all_age_wave5 = ((self.dataframe["IYEAR5"] + 1 / self.dataframe["IMONTH5"]) - (self.dataframe["H1GI1Y"] + 1 / self.dataframe["H1GI1M"])).mean() - 1900
        football_age_wave5 = ((football["IYEAR5"] + 1 / football["IMONTH5"]) - (
                    football["H1GI1Y"] + 1 / football["H1GI1M"])).mean() - 1900
        no_football_age_wave5 = ((no_football["IYEAR5"] + 1 / no_football["IMONTH5"]) - (
                    no_football["H1GI1Y"] + 1 / no_football["H1GI1M"])).mean() - 1900

        # stds
        std_all_age_wave5 = ((self.dataframe["IYEAR5"] + 1 / self.dataframe["IMONTH5"]) - (self.dataframe["H1GI1Y"] + 1 / self.dataframe["H1GI1M"])).std()
        std_football_age_wave5 = ((football["IYEAR5"] + 1 / football["IMONTH5"]) - (
                    football["H1GI1Y"] + 1 / football["H1GI1M"])).std()
        std_no_football_age_wave5 = ((no_football["IYEAR5"] + 1 / no_football["IMONTH5"]) - (
                    no_football["H1GI1Y"] + 1 / no_football["H1GI1M"])).std()

        # range
        range1_all_min = np.floor((self.dataframe["IYEAR5"] + 1 / self.dataframe["IMONTH5"]) - (self.dataframe["H1GI1Y"] + 1 / self.dataframe["H1GI1M"])).min() - 1900
        range1_all_max = np.floor((self.dataframe["IYEAR5"] + 1 / self.dataframe["IMONTH5"]) - (self.dataframe["H1GI1Y"] + 1 / self.dataframe["H1GI1M"])).max() - 1900
        range1_football_min = np.floor(
            (football["IYEAR5"] + 1 / football["IMONTH5"]) - (football["H1GI1Y"] + 1 / football["H1GI1M"])).min() - 1900
        range1_football_max = np.floor(
            (football["IYEAR5"] + 1 / football["IMONTH5"]) - (football["H1GI1Y"] + 1 / football["H1GI1M"])).max() - 1900
        range1_no_football_min = np.floor((no_football["IYEAR5"] + 1 / no_football["IMONTH5"]) - (
                    no_football["H1GI1Y"] + 1 / no_football["H1GI1M"])).min() - 1900
        range1_no_football_max = np.floor((no_football["IYEAR5"] + 1 / no_football["IMONTH5"]) - (
                    no_football["H1GI1Y"] + 1 / no_football["H1GI1M"])).max() - 1900

        # interquartile range
        range_q_all_min, range_q_all_max = np.percentile(
            np.floor((self.dataframe["IYEAR5"] + 1 / self.dataframe["IMONTH5"]) - (self.dataframe["H1GI1Y"] + 1 / self.dataframe["H1GI1M"])), [25, 75]) - 1900
        range_q_football_min, range_q_football_max = np.percentile(
            np.floor((football["IYEAR5"] + 1 / football["IMONTH5"]) - (football["H1GI1Y"] + 1 / football["H1GI1M"])),
            [25, 75]) - 1900
        range_q_no_football_min, range_q_no_football_max = np.percentile(np.floor(
            (no_football["IYEAR5"] + 1 / no_football["IMONTH5"]) - (no_football["H1GI1Y"] + 1 / no_football["H1GI1M"])),
                                                                         [25, 75]) - 1900

        # medians
        med_all_age_wave5 = np.floor(
            (self.dataframe["IYEAR5"] + 1 / self.dataframe["IMONTH5"]) - (self.dataframe["H1GI1Y"] + 1 / self.dataframe["H1GI1M"])).median() - 1900
        med_football_age_wave5 = np.floor((football["IYEAR5"] + 1 / football["IMONTH5"]) - (
                    football["H1GI1Y"] + 1 / football["H1GI1M"])).median() - 1900
        med_no_football_age_wave5 = np.floor((no_football["IYEAR5"] + 1 / no_football["IMONTH5"]) - (
                    no_football["H1GI1Y"] + 1 / no_football["H1GI1M"])).median() - 1900

        print("================================= Age Wave5  =================================")
        print(f"Mean               : {all_age_wave5} | {football_age_wave5} | {no_football_age_wave5}")
        print(f"Std                : {std_all_age_wave5} | {std_football_age_wave5} | {std_no_football_age_wave5}")
        print(f"Median             : {med_all_age_wave5} | {med_football_age_wave5} | {med_no_football_age_wave5}")
        print(
            f"Interquartile Range: {range_q_all_min}-{range_q_all_max} | {range_q_football_min}-{range_q_football_max} | {range_q_no_football_min}-{range_q_no_football_max}")
        print(
            f"Range              : {range1_all_min}-{range1_all_max} | {range1_football_min}-{range1_football_max} | {range1_no_football_min}-{range1_no_football_max}")


if __name__ == '__main__':
    paper = Iverson22Football()
    for find in paper.FINDINGS:
        print(find.run())
    paper.figure_finding()
    paper.table_finding()
