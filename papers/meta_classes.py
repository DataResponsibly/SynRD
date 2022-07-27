import json
import pandas as pd

class Publication():
    """
    A class wrapper for all publication classes, for shared functionality.
    """
    DEFAULT_PAPER_ATTRIBUTES = {
        'length_pages': 0,
        'authors': [],
        'journal': None,
        'year': 0,
        'current_citations': 0
    }

    FINDINGS = []

    DATAFRAME_COLUMNS = []

    FILENAME = None

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def run_all_findings(self):
        results = {}
        for finding in self.FINDINGS:
            result = finding.run()
            results[str(finding)] = result
        return results

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

class Finding():
    """
    A class wrapper for all findings, for shared functionality.
    """
    def __init__(self, 
                 finding_lambda, 
                 args=None, 
                 description=None, 
                 text=None):
        self.finding_lambda = finding_lambda
        self.args = args
        self.description = description
        self.text = text

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