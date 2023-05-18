from SynRD.synthesizers import MSTSynthesizer, PATECTGAN, PrivBayes, PacSynth

import numpy as np
from sklearn import datasets
iris = datasets.load_iris(as_frame=True)['data']
iris = iris.astype(int)

class TestSynthesizers:

    def test_synths(self):
        for s in [MSTSynthesizer, PATECTGAN, PrivBayes]:
            synth = s(epsilon=1.0, slide_range=False)
            synth.fit(iris)
            df = synth.sample(100)
            assert len(df) == 100

            # Check slide range
            synth = s(epsilon=1.0, slide_range=True)
            iris_slid = iris + 10
            synth.fit(iris_slid)
            df = synth.sample(100)
            assert len(df) == 100
    
    def test_pacsynth(self):
        # PacSynth is weird...
        synth = PacSynth(epsilon=1.0, slide_range=False)
        synth.fit(iris)
        df = synth.sample(100)

        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        df = df.fillna(0)
        assert len(df) == 100