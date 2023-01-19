# import SynRD
# from SynRD.papers import get_papers
import pandas as pd

from SynRD.synthesizers import MSTSynthesizer
from SynRD.benchmark import Benchmark
from SynRD.papers import Saw2018Cross


# Custom Synthesizer
class Synthesizer:
    def __init__(self) -> None:
        pass

    def fit(self, paper: pd.DataFrame):
        pass

    def sample(self, n):
        pass

    # etc.


def main():
    new_synth = Synthesizer()
    # dataset = load('original')

    benchmark = Benchmark()
    new_synth = MSTSynthesizer(epsilon=1.0, slide_range=False)
    saw = Saw2018Cross()
    for paper in [saw]: # need to make sure get_papers() is working
        print(str(paper))
        new_synth.fit(paper.real_dataframe)
        # dataset = new_synth.sample(len(paper))
        dataset = new_synth.sample(len(paper.real_dataframe))
        # Probably we could use property
        paper.set_synthetic_dataframe(dataset)
        benchmark.eval(paper)
        # benchmark.plot(plots=all)


    print(benchmark.summary())


if __name__ == "__main__":
    main()
