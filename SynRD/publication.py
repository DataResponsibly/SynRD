import json
import pandas as pd

from enum import Enum

class _MEAN_DIFFERENCE(Enum):
    # We are comparing apples before to apples now
    TEMPORAL_FIXED_CLASS = 1
    # We are comparing apples to oranges
    BETWEEN_CLASS = 2

class _PATH_ANALYSIS(Enum):
    # Degree of relationship between an apple and orange
    VARIABILITY = 1
    # Difference in orange or apple specific per unit coefficient change in pathway analysis
    COEFFICIENT_DIFFERENCE = 2
    # Dependence of an apple on the value of an orange
    INTERACTION_EFFECT = 3

class _LOGISTIC(Enum):
    # Overall, how well can we classify apples and/or oranges
    ACCURACY = 1
    # How many negative apples did we classify as positive
    FNR = 2
    # How many positive apples did we classify as negative
    FPR = 3
    # What proportion of all apples do we think are positive
    PREDICTED_BASE_RATE = 4

class _REGRESSION(Enum):
    # Structural equation modeling for examining multiple causal pathways between fruit
    # See _PATH_ANALYSIS
    PATH_ANALYSIS = _PATH_ANALYSIS
    # Difference in coefficient for OLS or other simple regression between apples and oranges
    COEFFICIENT_COMPARISON = 1
    # Just the apple coefficient sign
    COEFFICIENT_SIGN = 2
    # Classification of apples and oranges
    LOGISTIC = _LOGISTIC

class _CORRELATION(Enum):
    # Linear bivariate strength and direction of relationship between apples and oranges
    PEARSON_CORRELATION = 1
    # Non-linear relationship between apples and oranges
    # (usually used for ordinal or categorical data)
    SPEARMAN_CORRELATION = 2

class TAXONOMY(Enum):
    # See _MEAN_DIFFERENCE
    MEAN_DIFFERENCE = _MEAN_DIFFERENCE
    # Just describing apples or oranges or both
    DESCRIPTIVE_STATISTICS = 2
    # See _REGRESSION
    REGRESSION = _REGRESSION
    # See _CORRELATION
    CORRELATION = _CORRELATION


class Publication():
    """
    A class wrapper for all publication classes, for shared functionality.
    """
    cont_features = None
    DEFAULT_PAPER_ATTRIBUTES = {
        'id': None,
        'length_pages': 0,
        'authors': [],
        'journal': None,
        'year': 0,
        'current_citations': 0,
        'base_dataframe_pickle': None
    }

    FINDINGS = []

    DATAFRAME_COLUMNS = []

    FILENAME = None

    def __init__(self, dataframe=None, description=None):
        if dataframe is not None:
            self.dataframe = dataframe
            self.real_dataframe = dataframe
        else:
            raise ValueError("Must set dataframe to initialize a paper class.")
        
        self._description = description
        self.columns = self.real_dataframe.columns

    def run_all_findings(self):
        results = {}
        for finding in self.FINDINGS:
            result = finding.run()
            results[str(finding)] = result
        return results
    
    def run_all_non_visual_findings(self):
        results = {}
        for finding in self.FINDINGS:
            if (not isinstance(finding,VisualFinding)) and (not isinstance(finding,FigureFinding)):
                result = finding.run()
                results[str(finding)] = result
        return results

    def set_synthetic_dataframe(self, df):
        assert len(df.columns) > 0 
        assert set(df.columns) == set(self.columns)
        self.synthetic_dataframe = df

    def _read_pickle_dataframe(self):
        return pd.read_pickle(self.FILENAME + '_dataframe.pickle')

    def _recreate_dataframe(self):
        """
        Method for recreating a base dataframe for the publication.
        """
        raise NotImplementedError()

    def _validate_dataframe(self):
        return set(self.DATAFRAME_COLUMNS) == set(self.dataframe)

    def __str__(self):
        return json.dumps(self.DEFAULT_PAPER_ATTRIBUTES, sort_keys=True, indent=4)

    @property
    def description(self):
        if self._description:
            return self.description
        paper_info = self.DEFAULT_PAPER_ATTRIBUTES
        text = f'{paper_info["id"]}, authors: {", ".join(paper_info["authors"])}, year: {paper_info["year"]}\n'
        for find in self.FINDINGS:
            text += find.text + '(' + find.description + ')' + '\n'
        return text

class Finding():
    """
    A class wrapper for all findings, for shared functionality.
    """
    def __init__(self, 
                 finding_lambda, 
                 args=None, 
                 description=None, 
                 text=None,
                 finding_type: TAXONOMY = None):
        self.finding_lambda = finding_lambda
        self.args = args
        self.description = description
        self.text = text
        self.finding_type = finding_type

    def run(self):
        if self.args is None:
            return self.finding_lambda()
        return self.finding_lambda(**self.args)
    
    def __str__(self):
        return self.description

class VisualFinding(Finding):
    """
    A class wrapper for all findings, for shared functionality.
    """
    pass

class FigureFinding(Finding):
    """
    A class wrapper for all findings, for shared functionality.
    """
    pass

class NonReproducibleFindingException(Exception):
    pass
