class Benchmark:
    def __init__(self) -> None:
        self.results = {}

    def eval(self, paper):
        self.results["paper_name"] = "paper_results"

    def summary(self) -> str:
        return "\n".join(
            [f"{paper_name=}: {result=}" for paper_name, result in self.results.items()]
        )
