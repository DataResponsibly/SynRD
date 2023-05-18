import pandas as pd
from file_utils import PathSearcher
from meta_classes import Publication

# import sys
# sys.path.append('.')


class Liu2020Protective(Publication):
    """A class for the 'Protective factors against juvenile delinquency: Exploring gender'
    paper."""

    DEFAULT_PAPER_ATTRIBUTES = {
        "id": "liu2020protective",
        "length_pages": 12,
        "authors": ["Lin Liu", "Susan L. Miller"],
        "journal": "Social Science Research",
        "year": 2020,
        "current_citations": None,
        "base_dataframe_pickle": "liu2020protective_dataframe.pickle",
    }

    DATAFRAME_COLUMNS = [
        "S2",
        "S1",
        "H1GI18",
        "H1DS5",
        "H1FV7",
        "H1FV8",
        "H1FV9",
        "H1DS3",
        "H1DS4",
        "H1DS9",
        "H1DS13",
        "H1WP9",
        "H1WP13",
        "H1WP10",
        "H1WP14",
        "S62B",
        "S62I",
        "S62L",
        "S62E",
        "H1TO9",
        "H1TO29",
        "H1TO33",
        "H1WP1",
        "H1WP6",
        "H1WP2",
        "H1WP3",
        "H1WP7",
        "H1WP4",
        "H1WP5",
        "S10A",
        "S10B",
        "S10C",
        "S10D",
    ]

    FILENAME = "liu2020protective"

    def __init__(self, dataframe=None, filename=None, path=None):
        if path is None:
            path = self.FILENAME
        self.path_searcher = PathSearcher(path)

        if filename is not None:
            self.dataframe = pd.read_pickle(self.path_searcher.get_path(filename))
        elif dataframe is not None:
            self.dataframe = dataframe
        else:
            self.dataframe = self._recreate_dataframe()

        self.FINDINGS = self.FINDINGS + []

        # self.table_2_npmatrix = None
        #
        # self.table_2_structure = {
        #     "model_summary_subset_columns": ["R", "Rsq", "MSE", "F", "df1", "df2", "p"],
        #     "model_subset_columns": ["B", "se", "t", "p", "LLCI", "ULCI"],
        #     "uncond_effects_subset_columns": ["R2-chng", "F", "df1", "df2", "p"],
        #     "cond_effects_subset_columns": [
        #         "value",
        #         "effect",
        #         "se",
        #         "t",
        #         "p",
        #         "LLCI",
        #         "ULCI",
        #     ],
        # }
        # self.table_2_structure["response_rows"] = [
        #     ("model_summary", self.table_2_structure["model_summary_subset_columns"]),
        #     ("constant", self.table_2_structure["model_subset_columns"]),
        #     ("MENTOR", self.table_2_structure["model_subset_columns"]),
        #     ("PARENT_NO_EDU", self.table_2_structure["model_subset_columns"]),
        #     ("Int_1", self.table_2_structure["model_subset_columns"]),
        #     ("AGE_YEARS", self.table_2_structure["model_subset_columns"]),
        #     ("BIO_SEX", self.table_2_structure["model_subset_columns"]),
        #     ("RACE_HISPANIC", self.table_2_structure["model_subset_columns"]),
        #     ("RACE_BLACK", self.table_2_structure["model_subset_columns"]),
        #     ("RACE_OTHER", self.table_2_structure["model_subset_columns"]),
        #     ("uncond_effects", self.table_2_structure["uncond_effects_subset_columns"]),
        #     ("PARENT_NO_EDU==0", self.table_2_structure["cond_effects_subset_columns"]),
        #     ("PARENT_NO_EDU==1", self.table_2_structure["cond_effects_subset_columns"]),
        # ]
        # self.table_2_structure["response_column_subsets"] = [
        #     self.table_2_structure["model_summary_subset_columns"],
        #     self.table_2_structure["model_subset_columns"],
        #     self.table_2_structure["uncond_effects_subset_columns"],
        #     self.table_2_structure["cond_effects_subset_columns"],
        # ]

    def _filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # We excluded 128 participants who either refused to answer or were not
        # enrolled in school at the year of the survey, because one of our
        # research interests was to examine how young students’ social bonds
        # with their school protected them from delinquency.
        df = df[df["H1GI18"] == 1]
        df = df.drop("H1GI18", axis="columns")
        return df

    def _recreate_dataframe(
        self, filename="liu2020protective_dataframe.pickle", print_debug=False
    ):
        df = pd.read_csv(self.path_searcher.get_path("wave1.csv"))
        df = self._filter_dataframe(df)

        # wave1, wave3, wave4 = self._transform_separate_dataframes(
        #     wave1, wave3, wave4, print_debug
        # )
        # merged_wave_all = self._transform_merge_dataframes(
        #     wave1, wave3, wave4, print_debug
        # )
        # merged_wave_all_clean = self._transform_clean_merged_dataframe(
        #     merged_wave_all, print_debug
        # )

        # merged_wave_all_clean.to_pickle(filename)
        # return merged_wave_all_clean

    # def _validate_table_2(self):
    #     assert self.table_2_npmatrix.shape[0] == len(
    #         self.table_2_structure["response_rows"]
    #     )
    #     assert self.table_2_npmatrix.shape[1] == max(
    #         [len(cols) for cols in self.table_2_structure["response_column_subsets"]]
    #     )
    #
    #     for row in range(self.table_2_npmatrix.shape[0]):
    #         for col in range(self.table_2_npmatrix.shape[1]):
    #             if col >= len(self.table_2_structure["response_rows"][row][1]):
    #                 assert np.isnan(self.table_2_npmatrix[row, col])
    #
    # def _calculate_table_2(self, print_debug=False):
    #     r.source(self.path_searcher.get_path("processv41/PROCESS v4.1 for R/process.R"))
    #
    #     with localconverter(default_converter + pandas2ri.converter):
    #         r_df = py2rpy(self.dataframe)
    #
    #     r_covariate_names = vectors.StrVector(
    #         ["AGE_YEARS", "BIO_SEX", "RACE_HISPANIC", "RACE_BLACK", "RACE_OTHER"]
    #     )
    #
    #     r_response = r["process"](
    #         y="EDU_ATTAINED",
    #         x="MENTOR",
    #         w="PARENT_NO_EDU",
    #         cov=r_covariate_names,
    #         data=r_df,
    #         model=1,
    #         save=2,
    #     )
    #
    #     with localconverter(default_converter + pandas2ri.converter):
    #         py_response = rpy2py(r_response)
    #
    #     self.table_2_npmatrix = py_response
    #     self._validate_table_2()
    #     return py_response
    #
    # def table_2_check(self):
    #     # Unfortunately, the output is returned in a label-less matrix that is dependent on how the output
    #     # is printed from R. This appears to be stable when process method signature is exactly the same, but
    #     # this is a risky assumption. At least assert that response shape matches size of below.
    #
    #     if self.table_2_npmatrix is None:
    #         results = self._calculate_table_2()
    #         self.table_2_npmatrix = results
    #     else:
    #         results = self.table_2_npmatrix
    #
    #     return results
    #
    # def table_2(self):
    #     # TODO: Add formatting for table
    #     return self.table_2_check()
    #
    # def get_table_2_statistic(self, index, statistic):
    #     """Retrieves correct value from response matrix."""
    #     self.table_2_check()
    #     for i, (rr_index, rr_colname) in enumerate(
    #         self.table_2_structure["response_rows"]
    #     ):
    #         if rr_index == index:
    #             break
    #     for j, rc_stat in enumerate(rr_colname):
    #         if rc_stat == statistic:
    #             break
    #     return self.table_2_npmatrix[i, j]
    #
    # def finding_390_1(self):
    #     """The overall model was significant, F (8, 4172) = 111.98, p < .001,
    #     and explained approximately 18% of the variability in educational
    #     attainment (see Table 2)."""
    #     model_F = self.get_table_2_statistic("Model", "F")
    #     model_p = self.get_table_2_statistic("Model", "p")
    #
    #     values = [model_F, model_p]
    #     soft_finding = model_p < 0.05
    #     hard_findings = [model_F, model_p]
    #
    #     return (values, soft_finding, hard_findings)
    #
    # def finding_390_2(self):
    #     """As hypothesized, in the final model, we found a
    #     significant main effect of parental education (B = -1.91,
    #     t = 14.37 p < .001, 95% CI = -2.18 to -1.65) such
    #     that students who had at least one parent who graduated
    #     from college moved about two steps further in their education
    #     (e.g., the difference between a high school graduate
    #     and someone who completed a vocational certification, or
    #     the difference between a bachelors and doctoral degree)."""
    #     var_B = self.get_table_2_statistic("PARENT_NO_EDU", "B")
    #     var_p = self.get_table_2_statistic("PARENT_NO_EDU", "p")
    #
    #     values = [var_B, var_p]
    #     soft_finding = var_p < 0.05
    #     hard_findings = [var_B, var_p]
    #
    #     return (values, soft_finding, hard_findings)
    #
    # def finding_390_3(self):
    #     """There was also a significant main effect of the presence
    #     of a mentor (B = .40, t = 3.68 p < .001, 95% CI = .19
    #     to .61) such that students who reported having a mentor
    #     in adolescence or emerging adulthood, had significantly
    #     higher educational attainment than those who did not."""
    #     var_B = self.get_table_2_statistic("MENTOR", "B")
    #     var_p = self.get_table_2_statistic("MENTOR", "p")
    #
    #     values = [var_B, var_p]
    #     soft_finding = var_p < 0.05
    #     hard_findings = [var_B, var_p]
    #
    #     return (values, soft_finding, hard_findings)
    #
    # def finding_391_1(self):
    #     """Finally, there was a significant effect
    #     of race such that African Americans had significantly
    #     lower educational attainment than other participants."""
    #     var_B = self.get_table_2_statistic("RACE_BLACK", "B")
    #     var_p = self.get_table_2_statistic("RACE_BLACK", "p")
    #
    #     values = [var_B, var_p]
    #     soft_finding = var_p < 0.05
    #     hard_findings = [var_B, var_p]
    #
    #     return (values, soft_finding, hard_findings)
    #
    # def finding_391_2(self):
    #     """In addition, the analysis revealed a significant interaction
    #     between parental education and availability of a mentor in the
    #     final model, suggesting that having a mentor moderates the
    #     relationship between having a parent who graduated from college
    #     and educational attainment in adulthood. The interaction effect
    #     was very small but statistically significant (dR2 = .001, p < .05)."""
    #     var_B = self.get_table_2_statistic("Int_1", "B")
    #     var_p = self.get_table_2_statistic("Int_1", "p")
    #
    #     values = [var_B, var_p]
    #     soft_finding = var_p < 0.05
    #     hard_findings = [var_B, var_p]
    #
    #     return (values, soft_finding, hard_findings)
    #
    # def finding_391_3(self):
    #     """Having a mentor was more beneficial to young people whose parents did
    #     not attend college (B = .74, p < .001) than for those with at least one
    #     parent who is a college graduate (B = .40, p < .001)."""
    #     var_0_effect = self.get_table_2_statistic("PARENT_NO_EDU==0", "effect")
    #     var_0_p = self.get_table_2_statistic("PARENT_NO_EDU==0", "p")
    #     var_1_effect = self.get_table_2_statistic("PARENT_NO_EDU==1", "effect")
    #     var_1_p = self.get_table_2_statistic("PARENT_NO_EDU==1", "p")
    #
    #     # Implicit in the statement of finding is that both are significant
    #     values = [var_0_effect, var_0_p, var_1_effect, var_1_p]
    #     soft_finding = var_0_p < 0.05 and var_0_p < 0.05 and var_1_effect > var_0_effect
    #     hard_findings = [var_0_p, var_1_p, var_1_effect - var_0_effect]
    #
    #     return (values, soft_finding, hard_findings)
