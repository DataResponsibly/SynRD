import pandas as pd

from SynRD.publication import Publication


class Yuze23Marital(Publication):
    INPUT_FILES = ["yuze23marital/04690-0001-Data.tsv"]

    # The sample included 777 married women and 615 married men who completed study
    # measures at both waves.
    GENERAL_INFO = {
        "V1801": "Sex 1",
        "V5801": "Sex 2",
        "V2000": "Age",
        "V2102": "Race",
        "V4005": "Year of W2",
        "V9003": "Dead by end W2",
        # "V2060": "Married",
        # "V601": "Married",
        # "V4501": "Married",
        # "V2015": "MARRIED/NOTMARRIED",
        "V6060": "Married",
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
        # "V12152": "Marital satisfaction 1",
        # "V12153": "Feel loved 1",
        # "V12154": "Spouse listens 1",
        "V602": "Marital satisfaction 1",
        "V405": "Feel loved 1",
        "V407": "Spouse listens 1",
        # "V15409": "Marital satisfaction 2",
        # "V15410": "Fell loved 2",
        # "V15412": "Spouse listens 2",
        "V4502": "Marital satisfaction 2",
        "V4407": "Fell loved 2",
        "V4409": "Spouse listens 2",
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
        "V12472": "Was attacked",
        "V12468": "Was robbed",
        "V12482": "Had life-threatening illness",
        "V12484": "Has serious illness",
        "V12470": "Lost a job",
        "V12493": "Had financial problems",
        "V12474": "Parent died",
        "V12488": "Child died",
    }

    # Depressive symptoms were measured with the 11-item “Iowa” form (Kohout et al.,
    #                                                                     1993) of the
    # Center for Epidemiological Studies- Depression (CES– D; Radloff, 1977). Items
    # were rated on a 3-point rat- ing scale, reverse-scored as necessary so that
    # higher scores indicated greater levels of symptoms, and summed (α = 0.81 at Wave
    #                                                                 I and 0.82 at
    #                                                                 Wave II).
    DEPRESSIVE_SYMPTOMS = {
        "V2618": "CESD-11 1",
        "V6618": "CESD-11 2",
    }

    def __init__(self, dataframe=None):
        super().__init__(dataframe=dataframe)

    @classmethod
    def _recreate_dataframe(
        cls, filename="yuze23marital_dataframe.pickle"
    ) -> pd.DataFrame:
        assert len(cls.INPUT_FILES) == 1
        file_path = cls.INPUT_FILES[0]

        COLUMN_MAP = (
            cls.GENERAL_INFO
            | cls.MARITAL_SATISFACTION
            | cls.STRESSFUL_LIFE_EVENTS
            | cls.DEPRESSIVE_SYMPTOMS
        )
        df = pd.read_csv(file_path, sep="\t", usecols=COLUMN_MAP.keys())
        df = df.rename(columns=COLUMN_MAP)

        # This study focused on the Wave I survey, conducted in 1986 through
        # face-to-face interviews with 3617 participants in their homes by interviewers
        # from the University of Michigan’s Survey Research Center, which represents
        # 70 % of sampled households and 68 % of sampled individuals,
        assert len(df) == 3617

        # and the Wave II survey, conducted in 1989 through face-to-face interviews with
        # 2867 people interviewed in Wave I, which represents 83 % of those still alive
        # at the time.
        n_alive = len(df[df["Dead by end W2"] == 0])
        df = df[df["Year of W2"] == 1989]
        assert len(df) == 2867
        assert round(len(df) / n_alive, 2) == 0.83

        # The sample
        # included 777 married women and 615 married men who completed study measures at
        # both waves.
        df = df[df["Married"] == 1]
        # print(df.pivot_table(index='Married', columns='Dummy married', aggfunc='size', fill_value=0))
        for c in COLUMN_MAP.values():
            vc = df[c].value_counts()
            if "CESD" not in c:
                print(f"{c}: {vc[vc.index<=0].to_dict()}")
                df = df[df[c] >=0]

        print(
            df.pivot_table(
                index="Married", columns="Sex 1", aggfunc="size", fill_value=0
            )
        )

        # Participants had a mean age of 51.19 years (SD = 15.40), and the
        # mean length of marriage was 28.35 years (SD = 15.74).
        # The sample was 76 % White, 21 % Black, and 3 % other.
        # print(
        #     df.pivot_table(
        #         index="Married", columns="Dummy married", aggfunc="size", fill_value=0
        #     )
        # )
        # df = df[(df["Married"] == 1) & (df["Dummy married"] == 1)]
        # df.to_pickle(filename)
        return df


if __name__ == "__main__":
    df = Yuze23Marital._recreate_dataframe()
    # print(df)
    print(
        f"Expected: {{2: 777, 1: 615}}, actual: {df['Sex 1'].value_counts().to_dict()}"
    )
    print(f"Expected: 51.19, actual: {df['Age'].mean()}")