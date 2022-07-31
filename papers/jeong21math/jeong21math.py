from papers.meta_classes import Publication, Finding
import pandas as pd
import numpy as np


class Jeong2021Math(Publication):
    DEFAULT_PAPER_ATTRIBUTES = {
        'length_pages': 12,
        'authors': ['Haewon Jeong', 'Michael D. Wu', 'Nilanjana Dasgupta', 'Muriel Médard', 'Flavio P. Calmon'],
        'journal': None,
        'year': 2021,
        'current_citations': None
    }
    INPUT_COLUMNS = [
        'X1RACE', 'X1MTHID', 'X1MTHEFF', 'X1MTHINT', 'X1FAMINCOME', 'X1HHNUMBER', 'X1P1RELATION', 'X1PAR1EMP', 'X1PARRESP',
        'X1SCHOOLBEL', 'X1STU30OCC2', 'X1TXMSCR', 'S1M8GRADE', 'S1LANG1ST', 'S1TEPOPULAR', 'S1TEMAKEFUN', 'S1MTHCOMP',
        'S1SCICOMP', 'S1APCALC', 'S1IBCALC', 'S1MTCHVALUES', 'S1MTCHINTRST', 'S1MTCHFAIR', 'S1MTCHRESPCT', 'S1MTCHCONF',
        'S1MTCHEASY', 'X1PAR1EDU', 'X1PAR2EDU', 'X1PAR1OCC2', 'X1PAR2OCC2', 'P1REPEATGRD', 'P1ELLEVER', 'P1MARSTAT',
        'P1YRBORN1', 'P1YRBORN2', 'P1JOBNOW1', 'P1JOBONET1_STEM1', 'P1JOBONET2_STEM1', 'P1HHTIME', 'P1EDUASPIRE',
        'P1EDUEXPECT', 'P1MTHHWEFF', 'P1SCIHWEFF', 'P1ENGHWEFF', 'P1MTHCOMP', 'P1SCICOMP', 'P1ENGCOMP', 'P1MUSEUM',
        'P1COMPUTER'
    ]
    DATAFRAME_COLUMNS = []
    FILENAME = 'jeong2021math'

    def __init__(self, dataframe=None, filename=None):
        if filename is not None:
            dataframe = pd.read_pickle(filename)
        elif dataframe is not None:
            dataframe = dataframe
        else:
            dataframe = self._recreate_dataframe()
        super().__init__(dataframe)
        self.FINDINGS = self.FINDINGS + []

    def _recreate_dataframe(self, filename='jeong2021math_dataframe.pickle'):
        school_survey = pd.read_csv('data/36423-0001-Data.tsv', sep='\t')
        student_survey = pd.read_csv('data/36423-0002-Data.tsv', sep='\t')
        raise NotImplementedError()
