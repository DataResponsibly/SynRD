import pandas as pd

from SynRD.publication import Publication


class Yuze23Marital(Publication):
    INPUT_FILES = ["yuze23marital/04690-0001-Data.tsv"]

    # The sample included 777 married women and 615 married men who completed study
    # measures at both waves.
    GENERAL_INFO = {
        "V1801": "X1:SEX OF RESPONDENT",
        "V5801": "2X1:SEX OF RESPONDENT",
        "V2060": "Married",
        "V2015": "MARRIED/NOTMARRIED",
        "V6060": "Dummy married",
    }

    # 2.2.1. Marital satisfaction
    # Marital satisfaction was measured with three items:
    # (a) “Taking all things together, how satisfied are you with your
    # marriage/relation- ship?”,
    # (b) “How much does your (husband/wife/partner) make you feel loved and
    # cared for?”, and
    # (c) “How much is (he/she) willing to listen when you need to talk about
    # your worries or problems?” Items were rated on 5-point scales and summed
    # (α = 0.76 at Wave I, 0.79 at Wave II).
    MARITAL_SATISFACTION = {
        "V12152": "W4.C3.MarSatis.Marital Satisfaction",
        "V12153": "W4.C4.SpseCare.Spouse Makes R Feel Loved & Cared For",
        "V12154": "W4.C5.SpseLstn.Spouse Willing to Listen to Rs Worries/Problems",
        "V15410": " W5.SPSECARE.Spouse Makes R Feel Loved & Cared For",
        "V15409": " W5.MARSATIS.Marital Satisfaction",
        "V15412": " W5.SPSELSTN.Spouse Willing to Listen to Rs Worries/ Problems",
    }

    # 2.2.2. Stressful life events The assessment of stress was based on the
    # count (i.e., sum) of the number of eight dichotomously-coded stressful
    # life events occurring during the three years between Wave I and Wave II:
    # physical attack,
    # robbery or burglary,
    # life threatening illness,
    # serious illness,
    # job loss,
    # serious financial problem,
    # death of parent,
    # and death of child.
    # Because few individuals reported experiencing >3 stressful life events,
    # stressful life events were top-coded at 3 events.
    STRESSFUL_LIFE_EVENTS = {
        "V12472": "W4.N4.LEAttack.Have you been the victim of a serious physical attack or assault since last interview",
        "V12468": "W4.N2.LERob.Was R robbed or was Rs home burglarized since last interview",
        "V12482": "W4.N6.LELTIll.Have you had a life-threatening illness or accidental injury since last interview",
        "V12484": "W4.N7.LESerIl.Have you had any serious, but not life-threatening, illness or injury that occurred or got worse since last interview",
        "V12470": "W4.N3.LEJob.Have you involuntarily lost a job (other than for retirement) since last interview",
        "V12493": "W4.N11.LEFinPrb.Have you had any serious financial problems or difficulties since last interview",
        "V12474": "W4.N5.LEParDie.Has a parent or step-parent of yours died since last interview",
        "V12488": "W4.N9.ChildDie.Has a child of yours died since last interview",
    }

    # Depressive symptoms were measured with the 11-item “Iowa” form (Kohout et al.,
    #                                                                     1993) of the
    # Center for Epidemiological Studies- Depression (CES– D; Radloff, 1977). Items
    # were rated on a 3-point rat- ing scale, reverse-scored as necessary so that
    # higher scores indicated greater levels of symptoms, and summed (α = 0.81 at Wave
    #                                                                 I and 0.82 at
    #                                                                 Wave II).
    DEPRESSIVE_SYMPTOMS = {
        "V2618": "CESD-11, MEAN",
        "V6618": "W2:CESD-11, MEAN",
    }

    # {
    #     "V2102": "Race",
    #     "V103": "Gender",
    #     "V2000": "Age",
    #     "V2007": "Education",
    #     "V2020": "Income",
    #     "V2637": "Smoking",
    #     "V2623": "BMI",
    #     "V2681": "HTN",
    #     "V13214": "Exercise",
    #     "V2203": "Depressive symptoms",
    #     "V915": "Health",
    #     "V1860": "Weight",
    #     "V15003": "Response pattern",
    #     "V836": "Stroke wave 1",
    #     "V4838": "Stroke wave 2",
    #     "V10225": "Stroke wave 3",
    #     "V12305": "Stroke wave 4",
    #     "V15944": "Stroke wave 5",
    #     "V12302": "Any stroke",
    #     "V11036": (
    #         "W3.ACL Wave 3 Interview and Death Status(Dates of Death thru Aug 31, 1994).Revised. "
    #         "May_22_2014info/1=SlfRep/2=Proxy/3=NonResp/4=Dead",
    #     ),
    # }

    def __init__(self, dataframe=None):
        super().__init__(dataframe=dataframe)
        self.COLUMN_MAP = self.GENERAL_INFO | self.MARITAL_SATISFACTION

    @classmethod
    def _recreate_dataframe(cls, filename="yuze23marital_dataframe.pickle"):
        assert len(cls.INPUT_FILES) == 1
        file_path = cls.INPUT_FILES[0]
        # df = pd.read_csv(file_path, sep="\t", skipinitialspace=True)

        df = pd.read_csv(
            file_path, sep="\t", skipinitialspace=True, usecols=cls.COLUMN_MAP.keys()
        )
        df = df.rename(columns=cls.COLUMN_MAP)
        df = df[(df["Married"] == 1) & (df["Dummy married"] == 1)]
        return df
        # df = pd.read_csv(file_path, sep='\t', skipinitialspace=True, usecols=current_columns)


if __name__ == "__main__":
    df = Yuze23Marital._recreate_dataframe()
    print(df)
    print(
        f"Expected: {{2: 777, 1: 615}}, actual: {df['X1:SEX OF RESPONDENT'].value_counts().to_dict()}"
    )
    print(f"Expected: 51.19, actual: {df['Age'].mean()}")
