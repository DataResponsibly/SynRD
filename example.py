# import SynRD
# from SynRD.papers import get_papers
from SynRD import Benchmark, Publication, get_papers


# Custom Synthesizer
class Synthesizer:
    def __init__(self) -> None:
        pass

    def fit(self, paper: Publication):
        pass

    def sample(self, n):
        pass

    # etc.


def main():
    new_synth = Synthesizer()

    benchmark = Benchmark()
    for paper in get_papers():
        print(paper)
        new_synth.fit(paper)
        # dataset = new_synth.sample(len(paper))
        dataset = new_synth.sample(10)
        benchmark.eval(dataset)
        # benchmark.plot(plots=all)

    print(benchmark.summary())


if __name__ == "__main__":
    main()
