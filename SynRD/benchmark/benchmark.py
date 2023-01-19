import pandas as pd
import numpy as np

class Benchmark:
    def __init__(self) -> None:
        self.results = {}
        self.tests = [self.real_vs_private_soft_findings]

    def eval(self, paper, tests=None):
        results = {}
        tests_to_run = self.tests if tests is None else tests

        for test in tests_to_run:
            name, result = test(paper)
            results[name] = result

        self.results[paper.DEFAULT_PAPER_ATTRIBUTES['id']] = results

    def summary(self) -> str:
        return "\n".join(
            [f"{paper_name=}: {result=}" for paper_name, result in self.results.items()]
        )
    
    def real_vs_private_soft_findings(self, paper):
        """
        Calculate percent matching of soft findings on
        real vs private across epsilons and synthesizers
        """

        # Run all real non visual findings 
        paper.dataframe = paper.real_dataframe
        real_findings = paper.run_all_non_visual_findings()
        real_soft_findings = [result[1] for _, result in real_findings.items()]

        paper.dataframe = paper.synthetic_dataframe
        synth_findings = paper.run_all_non_visual_findings()
        synth_soft_findings = [result[1] for _, result in synth_findings.items()]

        # Compare
        percentage = sum([1 if r == s else 0 for r,s in zip(real_soft_findings,synth_soft_findings)]) / len(real_soft_findings)
        
        return 'Soft findings', percentage 

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

    