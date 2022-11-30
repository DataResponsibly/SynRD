class Benchmark:
    def __init__(self) -> None:
        self.results = {}

    def eval(self, paper):
        self.results["paper_name"] = "paper_results"

    def summary(self) -> str:
        return "\n".join(
            [f"{paper_name=}: {result=}" for paper_name, result in self.results.items()]
        )

    # TODO: Move functionality from plotting utils and publication aggregator to benchmark
    # TODO: Add confidence interval/bootstrapping functionality
    # TODO: Transition from pickles to CSVs (maybe not)
    

    # Andrii
    # TODO: Create sample for loading ICPSR zip -> csv -> dataframe using Andriis snippet
    # TODO: Transition from to lower level pickles
    
    # Stacy
    # TODO: Move synthesizers out from publication aggregator/private data generator
        # Involves creating separate wrapper classes
        # Involves figuring out how to deal with hyperparameters
            # SDV: https://github.com/sdv-dev/SDV/tree/master/sdv
        # Then, ideally, an example demonstrating that these classes compile and work

    