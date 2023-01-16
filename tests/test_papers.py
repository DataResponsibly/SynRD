from SynRD.papers import Saw2018Cross

class TestSaw2018Cross:

    def test_finding_526_1(self):
        saw2018cross = Saw2018Cross(filename='saw2018cross_dataframe.pickle')
        (_, soft_finding, [hard_finding]) = saw2018cross.finding_526_1()
        assert soft_finding == True
        assert hard_finding > 0.09

    def test_finding_526_1(self):
        pass