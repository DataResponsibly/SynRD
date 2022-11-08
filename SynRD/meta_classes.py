import json
import pandas as pd

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

    def __init__(self, dataframe=None, filename=None, description=None):
        self.dataframe = dataframe
        self._description = description

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

class NonReproducibleFindingException(Exception):
    pass
