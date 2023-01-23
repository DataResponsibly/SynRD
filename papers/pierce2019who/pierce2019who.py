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

            
            ]