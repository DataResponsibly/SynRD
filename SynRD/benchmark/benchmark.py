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

    def eval(self, paper, tests=None, verbose=False):
        results = {}
        tests_to_run = self.tests if tests is None else tests

        for test in tests_to_run:
            name, result = test(paper, verbose=verbose)
            results[name] = result
            
        self.results[paper.DEFAULT_PAPER_ATTRIBUTES['id']] = results
        return results
    
    def eval_soft_findings(self, paper, B, tests=None, verbose=False):
        """
        Assumes that samples = n * B, where n is the number of samples in the real dataset
        and B is the number of bootstrap samples.
        """
        assert(paper.synthetic_dataframe.shape[0] >= paper.real_dataframe.shape[0] * B)

        results = {}
        paper.dataframe = paper.real_dataframe
        real_results = self.soft_findings(paper, verbose=verbose)
        all_scores = []
        for b in range(B):
            paper.dataframe = paper.synthetic_dataframe.sample(n=paper.real_dataframe.shape[0], replace=True)
            synth_results = self.soft_findings(paper, verbose=verbose)
            score_real_vs_synth = self.score_real_vs_synth(real_results, synth_results)
            all_scores.append(score_real_vs_synth)
            
        self.results[paper.DEFAULT_PAPER_ATTRIBUTES['id']] = [np.mean(all_scores), np.std(all_scores), np.percentile(all_scores,[2.5,97.5])]
        return results
    
    def eval_soft_findings_each_finding(self, paper, B, tests=None, verbose=False):
        """
        Assumes that samples = n * B, where n is the number of samples in the real dataset
        and B is the number of bootstrap samples.
        """
        assert(paper.synthetic_dataframe.shape[0] >= paper.real_dataframe.shape[0] * B)
        
        paper.dataframe = paper.real_dataframe
        real_results = self.soft_findings(paper, verbose=verbose)
        all_scores = np.array([])
        for b in range(B):
            paper.dataframe = paper.synthetic_dataframe.sample(n=paper.real_dataframe.shape[0], replace=True)
            synth_results = self.soft_findings(paper, verbose=verbose)
            score_real_vs_synth_per_finding = self.score_real_vs_synth_per_finding(real_results, synth_results)
            if b == 0: 
                all_scores = np.array(score_real_vs_synth_per_finding)
            else:
                all_scores = np.vstack([all_scores, np.array(score_real_vs_synth_per_finding)])
        res = [np.mean(all_scores, axis=0), np.std(all_scores, axis=0), np.percentile(all_scores,[2.5,97.5], axis=0)]
        self.results[paper.DEFAULT_PAPER_ATTRIBUTES['id']] = res
        return res

    def summary(self) -> str:
        return "\n".join(
            [f"{paper_name}: {result}" for paper_name, result in self.results.items()]
        )
    
    def score_real_vs_synth(self, real_findings, synth_findings):
        """
        Calculate percent matching of soft findings on
        real vs private
        """
        return sum([1 if r == s else 0 for r,s in zip(real_findings,synth_findings)]) / len(real_findings)
    
    def compare_list(self, list1, list2):
        if len(list1) != len(list2):
            print('list not same length')
            return
        res = []
        for i in range(len(list1)):
            res.append(list1[i] == list2[i])
        return [int(i) for i in res]

    def score_real_vs_synth_per_finding(self, real_findings, synth_findings):
        return self.compare_list(real_findings, synth_findings)
    
    def soft_findings(self, paper, verbose=False):
        """
        Run soft findings on paper.dataframe
        """
        findings = paper.run_all_non_visual_findings()
        return [result[1] for _, result in findings.items()]
    
    def real_vs_private_soft_findings(self, paper, verbose=False):
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
        if verbose:
            print(real_soft_findings)
            print(synth_soft_findings)
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

    
