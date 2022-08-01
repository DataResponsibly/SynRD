import json
import pandas as pd

class PublicationAggregator():
    """
    Central class for aggregator functionality.
    
    Provides methods for:
    1. Generating results across publications comparing real to private
       performance

    2. Generating summary results over all publications (i.e. average 
       citations, etc.)
    
    3. Generating publication level results for single publications (works by
       default, where n_publications = 1)
    """

    def __init__(self, publications, ):
        self.dataframe = dataframe