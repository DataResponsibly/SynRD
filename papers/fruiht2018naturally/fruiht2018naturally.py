import numpy as np
import pandas as pd

from rpy2.robjects import default_converter, r, pandas2ri, vectors
from rpy2.robjects.conversion import localconverter, py2rpy, rpy2py
from rpy2.robjects.packages import importr

from file_utils import PathSearcher
from meta_classes import FigureFinding, Finding, Publication, VisualFinding


class Fruiht2018Naturally(Publication):
    """A class for the 'Naturally Occurring Mentorship
    in a National Sample of First-Generation College Goers:
    A Promising Portal for Academic and Developmental Success'
    paper."""

    DEFAULT_PAPER_ATTRIBUTES = {
        "id": "fruiht2018naturally",
        "length_pages": 19,
        "authors": ["Veronica Fruiht", "Thomas Chan"],
        "journal": "American Journal of Community Psychology",
        "year": 2018,
        "current_citations": 42,
        "base_dataframe_pickle": "fruiht2018naturally_dataframe.pickle"
    }

    DATAFRAME_COLUMNS = [
        "BIO_SEX", "AGE_YEARS", "MENTOR", "PARENT_NO_EDU", "RACE", 
        "RACE_HISPANIC", "RACE_WHITE", "RACE_BLACK", "RACE_OTHER",
        "EDU_ATTAINED", "EDU_ATTAINED_BINARY"
    ]

    FILENAME = "fruiht2018naturally"

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
    
        self.FINDINGS = self.FINDINGS + [
            # Note: Table 1, Figure 1, and several findings can not be replicated
            # due to the manual coding task undertaken. Omit or is there another way
            # to signal that this will not be done? Will comment out at bottom.
            VisualFinding(self.table_2, description="table_2"),
            Finding(self.finding_390_1, description="finding_390_1",
                    text="""The overall model was significant, F (8,
                            4172) = 111.98, p < .001, and explained approximately
                            18% of the variability in educational attainment (see
                            Table 2)."""),
            Finding(self.finding_390_2, description="finding_390_2",
                    text="""As hypothesized, in the final model, we found a
                            significant main effect of parental education (B = -1.91,
                            t = 14.37 p < .001, 95% CI = -2.18 to -1.65) such
                            that students who had at least one parent who graduated
                            from college moved about two steps further in their education 
                            (e.g., the difference between a high school graduate
                            and someone who completed a vocational certification, or
                            the difference between a bachelors and doctoral degree)."""),
            Finding(self.finding_390_3, description="finding_390_3",
                    text="""There was also a significant main effect of the presence
                            of a mentor (B = .40, t = 3.68 p < .001, 95% CI = .19
                            to .61) such that students who reported having a mentor
                            in adolescence or emerging adulthood, had significantly
                            higher educational attainment than those who did not."""),
            Finding(self.finding_391_1, description="finding_391_1",
                    text="""Finally, there was a significant effect
                            of race such that African Americans had significantly
                            lower educational attainment than other participants."""),
            Finding(self.finding_391_2, description="finding_391_2",
                    text="""In addition, the analysis revealed a significant interaction
                            between parental education and availability of a mentor in the
                            final model, suggesting that having a mentor moderates the 
                            relationship between having a parent who graduated from college 
                            and educational attainment in adulthood. The interaction effect 
                            was very small but statistically significant (dR2 = .001, p < .05)."""),
            Finding(self.finding_391_3, description="finding_391_3",
                    text="""Having a mentor was more beneficial to young people whose parents did 
                            not attend college (B = .74, p < .001) than for those with at least one 
                            parent who is a college graduate (B = .40, p < .001)."""),
            #VisualFinding(self.table_1, description="table_1"),
            #FigureFinding(self.figure_1, description="figure_1"),
            #Finding(self.finding_392_1, description="finding_392_1",
            #        text="""There were no significant differences between groups in socioemotional 
            #                supports (v2 (3) = 5.72, p = .126)."""),
            #Finding(self.finding_392_2, description="finding_392_2",
            #        text="""The least common mentoring functions were cognitive supports, as just 
            #                23.9% of individuals reported these supports."""),
            #Finding(self.finding_392_3, description="finding_392_3",
            #        text="""Although the overall test of independence was non-significant 
            #                (v2 (3) = 5.76, p = .124) ..."""),
            #Finding(self.finding_392_4, description="finding_392_4",
            #        text="""... between group comparisons demonstrated that 
            #                continuing-generation college goers received significantly more of these 
            #                supports from their mentors (25.3% reporting at least one) than young people
            #                who did not go to college but whose parents were college graduates 
            #                (19.1% reporting at least one)."""),
            #Finding(self.finding_392_5, description="finding_392_5",
            #        text="""However, neither group was significantly different in cognitive mentoring
            #                functions than FGC students (23.3% reporting at least one)."""),
            #Finding(self.finding_392_6, description="finding_392_6",
            #        text="""The most marked between group differences were in the realm of support for
            #                identity development (v2 (3) = 27.47, p < .001)."""),
            #Finding(self.finding_392_7, description="finding_392_7",
            #        text="""Overall, students who attended college received substantially more support 
            #                for identity development (38.0%) than did non-college goers (29.3%)."""),
            #Finding(self.finding_392_8, description="finding_392_8",
            #        text="""Neither FGC students nor non-college goers with college-graduate parents 
            #                differed significantly from non-college goers whose parents had attended 
            #                college (32.4%)."""),
            #Finding(self.finding_392_9, description="finding_392_9",
            #        text="""Continuing-generation students received significantly less tangible support 
            #                from their mentors than any other group (6.8%)."""),
            #Finding(self.finding_392_10, description="finding_392_10",
            #        text="""GC students (9.9%) received more than continuing generation, but significantly 
            #                less than non-college goers whose parents did not attend college (14.5%)."""),
            #Finding(self.finding_392_11, description="finding_392_11",
            #        text="""Neither group was significantly different from non-college goers with 
            #                college-graduate parents (11.9%)."""),
        ]

        self.table_2_npmatrix = None

        self.table_2_structure = {
            "model_summary_subset_columns":  ["R", "Rsq", "MSE", "F", "df1", "df2", "p"],
            "model_subset_columns": ["B", "se", "t", "p", "LLCI", "ULCI"],
            "uncond_effects_subset_columns": ["R2-chng", "F", "df1", "df2", "p"],
            "cond_effects_subset_columns": ["value", "effect", "se", "t", "p", "LLCI", "ULCI"]
        }
        self.table_2_structure["response_rows"] = [
                ("model_summary", self.table_2_structure["model_summary_subset_columns"]),
                ("constant", self.table_2_structure["model_subset_columns"]),
                ("MENTOR", self.table_2_structure["model_subset_columns"]),
                ("PARENT_NO_EDU", self.table_2_structure["model_subset_columns"]),
                ("Int_1", self.table_2_structure["model_subset_columns"]),
                ("AGE_YEARS", self.table_2_structure["model_subset_columns"]),
                ("BIO_SEX", self.table_2_structure["model_subset_columns"]),
                ("RACE_HISPANIC", self.table_2_structure["model_subset_columns"]),
                ("RACE_BLACK", self.table_2_structure["model_subset_columns"]),
                ("RACE_OTHER", self.table_2_structure["model_subset_columns"]),
                ("uncond_effects", self.table_2_structure["uncond_effects_subset_columns"]),
                ("PARENT_NO_EDU==0", self.table_2_structure["cond_effects_subset_columns"]),
                ("PARENT_NO_EDU==1", self.table_2_structure["cond_effects_subset_columns"])
        ]
        self.table_2_structure["response_column_subsets"] = [
                self.table_2_structure["model_summary_subset_columns"],
                self.table_2_structure["model_subset_columns"],
                self.table_2_structure["uncond_effects_subset_columns"],
                self.table_2_structure["cond_effects_subset_columns"]
        ]

    def _transform_separate_dataframes(self, wave1, wave3, wave4, print_debug=False):
        def transform_age(row):
            if row["H1GI1Y"] == 96 or row["H1GI1M"] == 96:
                # Participant did not share their birthdate, treat age as missing
                return 98
            else:
                age_in_years = row["IYEAR"] - row["H1GI1Y"]
                if row["IMONTH"] < row["H1GI1M"]:
                    # Haven't reached birth month of year in interview
                    age_in_years -= 1
            return int(age_in_years)

        def transform_parent_no_college(row):
            values = [row["H1NM4"], row["H1NF4"], row["H1RM1"], row["H1RF1"]]
            # At least one parent graduated college
            if 8 in values or 9 in values: return 0
            # All parents have error code, condense to one error code
            if min(values) >= 97: return 98
            # If not, code as no parental college education
            return 1
        
        def transform_participant_graduated(row):
            if row["H4ED2"] <= 4:
                return 0
            elif row["H4ED2"] >= 5 and row["H4ED2"] <= 13:
                return 1
            else:
                # Return missing value
                return 98

        def transform_race(row):
            if row["H1GI4"] == 1: 
                race = 0  # Latino
            elif row["H1GI8"] in [1, 2, 3, 4, 5]:
                race = row["H1GI8"]
            elif row["H1GI8"] == 7:
                if row["H1GI6A"] == 1: race = 1    # White
                elif row["H1GI6B"] == 1: race = 2  # Black
                elif row["H1GI6C"] == 1: race = 3  # Native American
                elif row["H1GI6D"] == 1: race = 4  # Asian / PI
                elif row["H1GI6E"] == 1: race = 5  # Other
                else: race = 98
            else:
                race = 98  # Mark all missing as "don't know" for convienience
            
            return race

        wave1["AGE_YEARS"] = wave1.apply(transform_age, axis=1)
        wave1["PARENT_NO_EDU"] = wave1.apply(transform_parent_no_college, axis=1)
        wave1["RACE"] = wave1.apply(transform_race, axis=1)
        wave1["RACE_HISPANIC"] = wave1["RACE"] == 0
        wave1["RACE_HISPANIC"] = wave1["RACE_HISPANIC"].astype(int)
        wave1["RACE_WHITE"] = wave1["RACE"] == 1
        wave1["RACE_WHITE"] = wave1["RACE_WHITE"].astype(int)
        wave1["RACE_BLACK"] = wave1["RACE"] == 2
        wave1["RACE_BLACK"] = wave1["RACE_BLACK"].astype(int)
        wave1["RACE_OTHER"] = ((wave1["RACE"] == 3) |
                            (wave1["RACE"] == 4) |
                            (wave1["RACE"] == 5))
        wave1["RACE_OTHER"] = wave1["RACE_OTHER"].astype(int)
        wave1["BIO_SEX"].replace(1, 0, inplace=True)  # male to 0
        wave1["BIO_SEX"].replace(2, 1, inplace=True)  # female to 1 
        wave3["MENTOR"] = wave3["H3MN1"]
        wave4["EDU_ATTAINED"] = wave4["H4ED2"]
        wave4["EDU_ATTAINED_BINARY"] = wave4.apply(transform_participant_graduated, 
                                                   axis=1)
    
        return wave1, wave3, wave4
    
    def _transform_merge_dataframes(self, wave1, wave3, wave4, print_debug=False):
        merged_wave_13 = wave1.merge(wave3, how="inner", on="AID", 
                                     validate="one_to_one")
        merged_wave_all = merged_wave_13.merge(wave4, how="inner", on="AID", 
                                               validate="one_to_one")
        
        if print_debug:
            print("Before (wave 1), number of rows:", wave1.shape[0])
            print("After (merged wave 1 & 3), number of rows:", 
                merged_wave_13.shape[0])
            print("Percentage reduction:", 
                (wave1.shape[0]-merged_wave_13.shape[0]) / wave1.shape[0])
            
            print("Before (merged wave 1 & 3), number of rows:", 
                merged_wave_13.shape[0])
            
            print("After (merged wave 1, 3 & 4), number of rows:", 
                merged_wave_all.shape[0])
            print("Percentage reduction:", (merged_wave_13.shape[0] -
                merged_wave_all.shape[0]) / merged_wave_13.shape[0])
            print("Percentage reduction (including reported 28 missing values):", 
                (merged_wave_13.shape[0]-merged_wave_all.shape[0]+28) / 
                merged_wave_13.shape[0])

        return merged_wave_all

    def _transform_clean_merged_dataframe(self, merged_wave_all, print_debug=False):
        drop_codes = {
            "BIO_SEX": [6],
            "MENTOR": [8],
            "PARENT_NO_EDU": [98],
            "RACE": [98],
            "EDU_ATTAINED": [98],
            "EDU_ATTAINED_BINARY": [98],
            "AGE_YEARS": [98]
        }

        def in_drop_codes(row):
            for col, val in row.items():
                if col in drop_codes and val in drop_codes[col]:
                    return True
            return False
        
        merged_wave_all_clean = merged_wave_all.loc[
                ~merged_wave_all.apply(in_drop_codes, axis=1), :]

        merged_wave_all_clean = merged_wave_all_clean[self.DATAFRAME_COLUMNS]
        
        if print_debug:
            print("Number of participants after cleaning:",
                  merged_wave_all_clean.shape[0])
            print("Number of dropped participants due to missing / non-response:",
                  merged_wave_all.shape[0] - merged_wave_all_clean.shape[0])
        
        return merged_wave_all_clean

    def _recreate_dataframe(self,
                            filename="fruiht2018naturally.pickle",
                            print_debug=False):
        wave1 = pd.read_csv(self.path_searcher.get_path("wave1.csv"))
        wave3 = pd.read_csv(self.path_searcher.get_path("wave3.csv"))
        wave4 = pd.read_csv(self.path_searcher.get_path("wave4.csv"))

        wave1, wave3, wave4 = self._transform_separate_dataframes(wave1, wave3, wave4, print_debug)    
        merged_wave_all = self._transform_merge_dataframes(wave1, wave3, wave4, print_debug)
        merged_wave_all_clean = self._transform_clean_merged_dataframe(merged_wave_all, print_debug)

        merged_wave_all_clean.to_pickle(filename)
        return merged_wave_all_clean

    def _validate_table_2(self):
        assert self.table_2_npmatrix.shape[0] == len(self.table_2_structure["response_rows"])
        assert self.table_2_npmatrix.shape[1] == max(
                [len(cols) for cols in self.table_2_structure["response_column_subsets"]])

        for row in range(self.table_2_npmatrix.shape[0]):
            for col in range(self.table_2_npmatrix.shape[1]):
                if col >= len(self.table_2_structure["response_rows"][row][1]):
                    assert np.isnan(self.table_2_npmatrix[row, col])

    def _calculate_table_2(self, print_debug=False):
        r.source(self.path_searcher.get_path('processv41/PROCESS v4.1 for R/process.R'))

        with localconverter(default_converter + pandas2ri.converter):
            r_df = py2rpy(self.dataframe)
        
        r_covariate_names = vectors.StrVector(["AGE_YEARS", "BIO_SEX", "RACE_HISPANIC", "RACE_BLACK", "RACE_OTHER"])

        r_response = r['process'](y="EDU_ATTAINED", x="MENTOR", w="PARENT_NO_EDU", cov=r_covariate_names,
                                  data=r_df, model=1, save=2)

        with localconverter(default_converter + pandas2ri.converter):
            py_response = rpy2py(r_response)
        
        self.table_2_npmatrix = py_response
        self._validate_table_2()
        return py_response

    def table_2_check(self):
        # Unfortunately, the output is returned in a label-less matrix that is dependent on how the output
        # is printed from R. This appears to be stable when process method signature is exactly the same, but
        # this is a risky assumption. At least assert that response shape matches size of below.

        if self.table_2_npmatrix is None:
            results = self._calculate_table_2()
            self.table_2_npmatrix = results
        else:
            results = self.table_2_npmatrix
        
        return results

    def table_2(self):
        # TODO: Add formatting for table
        return self.table_2_check()

    def get_table_2_statistic(self, index, statistic):
        """Retrieves correct value from response matrix."""
        self.table_2_check()
        for i, (rr_index, rr_colname) in enumerate(self.table_2_structure["response_rows"]):
            if rr_index == index: break
        for j, rc_stat in enumerate(rr_colname):
            if rc_stat == statistic: break
        return self.table_2_npmatrix[i, j]

    def finding_390_1(self):
        """The overall model was significant, F (8, 4172) = 111.98, p < .001, 
        and explained approximately 18% of the variability in educational 
        attainment (see Table 2)."""
        model_F = self.get_table_2_statistic("Model", "F")
        model_p = self.get_table_2_statistic("Model", "p")

        values = [model_F, model_p]
        soft_finding = model_p < 0.05
        hard_findings = [model_F, model_p]

        return (values, soft_finding, hard_findings)

    def finding_390_2(self):
        """As hypothesized, in the final model, we found a
        significant main effect of parental education (B = -1.91,
        t = 14.37 p < .001, 95% CI = -2.18 to -1.65) such
        that students who had at least one parent who graduated
        from college moved about two steps further in their education 
        (e.g., the difference between a high school graduate
        and someone who completed a vocational certification, or
        the difference between a bachelors and doctoral degree)."""
        var_B = self.get_table_2_statistic("PARENT_NO_EDU", "B")
        var_p = self.get_table_2_statistic("PARENT_NO_EDU", "p")

        values = [var_B, var_p]
        soft_finding = var_p < 0.05
        hard_findings = [var_B, var_p]

        return (values, soft_finding, hard_findings)
    
    def finding_390_3(self):
        """There was also a significant main effect of the presence
        of a mentor (B = .40, t = 3.68 p < .001, 95% CI = .19
        to .61) such that students who reported having a mentor
        in adolescence or emerging adulthood, had significantly
        higher educational attainment than those who did not."""
        var_B = self.get_table_2_statistic("MENTOR", "B")
        var_p = self.get_table_2_statistic("MENTOR", "p")

        values = [var_B, var_p]
        soft_finding = var_p < 0.05
        hard_findings = [var_B, var_p]

        return (values, soft_finding, hard_findings)
    
    def finding_391_1(self):
        """Finally, there was a significant effect
        of race such that African Americans had significantly
        lower educational attainment than other participants."""
        var_B = self.get_table_2_statistic("RACE_BLACK", "B")
        var_p = self.get_table_2_statistic("RACE_BLACK", "p")

        values = [var_B, var_p]
        soft_finding = var_p < 0.05
        hard_findings = [var_B, var_p]

        return (values, soft_finding, hard_findings)
    
    def finding_391_2(self):
        """In addition, the analysis revealed a significant interaction
        between parental education and availability of a mentor in the
        final model, suggesting that having a mentor moderates the 
        relationship between having a parent who graduated from college 
        and educational attainment in adulthood. The interaction effect 
        was very small but statistically significant (dR2 = .001, p < .05)."""
        var_B = self.get_table_2_statistic("Int_1", "B")
        var_p = self.get_table_2_statistic("Int_1", "p")

        values = [var_B, var_p]
        soft_finding = var_p < 0.05
        hard_findings = [var_B, var_p]

        return (values, soft_finding, hard_findings)
    
    def finding_391_3(self):
        """Having a mentor was more beneficial to young people whose parents did 
        not attend college (B = .74, p < .001) than for those with at least one 
        parent who is a college graduate (B = .40, p < .001)."""
        var_0_effect = self.get_table_2_statistic("PARENT_NO_EDU==0", "effect")
        var_0_p = self.get_table_2_statistic("PARENT_NO_EDU==0", "p")
        var_1_effect = self.get_table_2_statistic("PARENT_NO_EDU==1", "effect")
        var_1_p = self.get_table_2_statistic("PARENT_NO_EDU==1", "p")

        # Implicit in the statement of finding is that both are significant
        values = [var_0_effect, var_0_p, var_1_effect, var_1_p]
        soft_finding = var_0_p < 0.05 and var_0_p < 0.05 and var_1_effect > var_0_effect
        hard_findings = [var_0_p, var_1_p, var_1_effect - var_0_effect]

        return (values, soft_finding, hard_findings)

    
        