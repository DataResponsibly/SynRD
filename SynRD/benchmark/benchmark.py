import pandas as pd
import numpy as np

from SynRD.publication import Publication
from SynRD.datasets.dataset_loader import DataRetriever
from SynRD.utils import _class_to_papername

class Benchmark:
    def __init__(self) -> None:
        self.results = {}
        self.tests = [self.real_vs_private_soft_findings]

    def initialize_papers(self, papers: "list[Publication]"):
        df_map = DataRetriever(papers).retrieve_necessary_data()
        initialized_papers = []
        for paper in papers:
            real_dataframe = df_map[_class_to_papername(paper) + "_processed"]
            initialized_papers.append(paper(dataframe=real_dataframe))
        return initialized_papers

    def eval(self, paper, tests=None):
        results = {}
        tests_to_run = self.tests if tests is None else tests

        for test in tests_to_run:
            name, result = test(paper)
            results[name] = result

        self.results[paper.DEFAULT_PAPER_ATTRIBUTES['id']] = results

    def summary(self) -> str:
        return "\n".join(
            [f"{paper_name}: {result}" for paper_name, result in self.results.items()]
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

    
def plot_example():
    import random

    import pandas as pd
    import plotly.express as px

    values = list(range(6))
    values = "Nemo enim ipsam voluptatem quia voluptas ".split()

    cols = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do".replace(
        ",", ""
    ).split()

    data = [random.choices(values, k=100) for _ in range(len(cols))]

    df = pd.DataFrame(index=cols, data=data)

    df_ = (
        df.stack()
        .reset_index()
        .rename(columns={"level_0": "index", "level_1": "col", 0: "value"})
        .groupby(["index", "value"])["col"]
        .count() .reset_index(name="n")
        )

    df_["value"] = df_["value"].astype(str)

    px.bar(df_, y="index", x="n", color="value", orientation="h").show()

    
