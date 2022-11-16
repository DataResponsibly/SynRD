# import SynRD
# from SynRD.papers import get_papers
import pandas as pd

from SynRD import Benchmark, get_papers


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

    benchmark = Benchmark()
    for paper in get_papers():
        print(paper)
        new_synth.fit(paper.dataframe)
        # dataset = new_synth.sample(len(paper))
        dataset = new_synth.sample(10)
        # Probably we could use property
        paper.set_dataframe(dataset)
        benchmark.eval(paper)
        # benchmark.plot(plots=all)

    print(benchmark.summary())


if __name__ == "__main__":
    main()
