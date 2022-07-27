from meta_classes import Publication, Finding

import pandas as pd
import numpy as np

class Lee2021Ability(Publication):
    """
    A class wrapper for all publication classes, for shared functionality.
    """
    DEFAULT_PAPER_ATTRIBUTES = {
        'length_pages': 0,
        'authors': [],
        'journal': '',
        'year': 0,
        'current_citations': 0
    }

    def __init__(self, dataframe=None, filename=None):
        if filename is not None:
            self.dataframe = pd.read_pickle(filename)
        elif dataframe is not None:
            self.dataframe = dataframe
        else:
            self.dataframe = self._recreate_dataframe()
        
        self.FINDINGS = self.FINDINGS + [

        ]
    
    def _recreate_dataframe(self, filename='saw2018cross_dataframe.pickle'):
        pass