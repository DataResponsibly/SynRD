from synthesizer import Synthesizer
import pandas as pd
from models import GSD
from stats import ChainedStatistics, Marginals
from jax.random import KeyArray
from src.utils import Dataset, Domain
import numpy as np
from snsynth.transform.table import TableTransformer

class GsdSynthesizer(Synthesizer):
    """
    Genetic algorithm synthesizer.

    ----------
    Parameters
        epsilon : float
            Privacy budget for the synthesizer.
    -----------
    Optional keyword arguments:
        slide_range : bool = False
            Specifies if the slide range transformation should be applied, this will 
            make the minimal value of each column 0 before fitting.
        thresh : float = 0.05
            Specifies what the ratio of unique values to the column length should be for
            the column to be threated as cathegorical.
        delta : float = 1e-09
            Privacy parameter, should be small, in the range of 1/(n * sqrt(n)).
        num_generations : int = 20000
            Total number of generations to run algorithm for.
        data_size : int = 2000
            The size of resulting dataframe.
        population_size_muta : int = 50
            Mutations population size.
        population_size_cross: int = 50
            Crossover population size.
        population_size : int = None
            Total size of the population. If None will be taken as sum of mutations and crossover
            populations. Otherwise the sizes of mutations and crossover populations will be derived
            as a half of the total population size.
        muta_rate : int = 1
            The number of rows altered with the mutation operation.
        mate_rate : int = 1
            The number of rows altered with th crossover operation.
        print_progress : bool = False
            Specifies if additional information should be printed or not.
        stop_early : bool = True
            Specifies if early stopping mechanism should be applied.
        stop_early_gen : int = None
            The number of generations after which the early stopping is available. If this value
            is set to k, then early stopping condition will be checked on every k-th generation.
            Setting it to None will result in the same behaviour as setting it to the data_size
            value.
        stop_eary_threshold : float = 0
            Defines the minimal fitness score the generations best candidate should have to proceed
            with the algorithm.
        sparse_statistics : bool = False
            Defines if sparsed statistics should be used.

    """
    def __init__(
        self,
        epsilon: float = None,
        slide_range: bool = None,
        thresh: float = None,
        delta: float = None, 
        num_generations : int = None,
        data_size : int = None,
        population_size_muta : int = None,
        population_size_cross: int = None,
        population_size : int = None,
        muta_rate : int = None,
        mate_rate : int = None,
        print_progress : bool = None,
        stop_early : bool = None,
        stop_early_gen : int = None,
        stop_early_threshold : float = None,
        sparse_statistics=False,
        **synth_kwargs: dict()
    ):
        super().__init__(epsilon, slide_range, thresh)
        allowed_additional_params = {"delta", "num_generations", "data_size", "population_size_muta",
                                     "population_size_cross", "population_size", "muta_rate",
                                     "mate_rate", "print_progress", "stop_early", "stop_early_gen",
                                     "stop_early_threshold", "sparse_statistics"}
        for param in synth_kwargs.keys():
            if param not in allowed_additional_params:
                raise ValueError(
                    f"Parameter '{param}' is not available for this type of synthesizer."
                )

        param_defaults = {
            "delta": (float, 1e-9), 
            "num_generations": (int, 20000), 
            "data_size": (int, 2000), 
            "population_size_muta": (int, 50),
            "population_size_cross": (int, 50), 
            "population_size": (int, None), 
            "muta_rate": (int, 1),
            "mate_rate": (int, 1), 
            "print_progress": (bool, False), 
            "stop_early": (bool, True), 
            "stop_early_gen": (int, None),
            "stop_early_threshold": (float, 0), 
            "sparse_statistics": (bool, False)
        }

        for param, (param_type, default_value) in param_defaults.items():
            param_value = locals().get(param)
            if param_value is not None:
                if type(param_value) is int and param_type is float:
                    param_value = float(param_value)
                if isinstance(param_type, tuple):
                    correctly_typed = False
                    for single_type in param_type:
                        if type(param_value) is single_type:
                            correctly_typed = True
                    if not correctly_typed:
                        raise TypeError(
                        f"{param} must be of one of the types {', '.join(list(map(lambda x: x.__name__, param_type)))}, got {type(param_value).__name__}."
                    )
                elif type(param_value) is not param_type:
                    raise TypeError(
                        f"{param} must be of type {param_type.__name__}, got {type(param_value).__name__}."
                    )
                setattr(self, param, param_value)
            else:
                setattr(self, param, default_value)

        self.synth_kwargs = synth_kwargs
        # self.synthesizer = SmartnoiseAIMSynthesizer(
        #     epsilon=self.epsilon,
        #     delta=self.delta,
        #     max_model_size=self.max_model_size,
        #     degree=self.degree,
        #     num_marginals=self.num_marginals,
        #     max_cells=self.max_cells,
        #     rounds=self.rounds,
        #     verbose=self.verbose,
        #     **synth_kwargs,
        # )
    def _get_train_data(self, data, *ignore, style, transformer, categorical_columns, ordinal_columns, continuous_columns, nullable, preprocessor_eps):
        if transformer is None or isinstance(transformer, dict):
            self._transformer = TableTransformer.create(data, style=style,
                categorical_columns=categorical_columns,
                continuous_columns=continuous_columns,
                ordinal_columns=ordinal_columns,
                nullable=nullable,
                constraints=transformer)
        elif isinstance(transformer, TableTransformer):
            self._transformer = transformer
        else:
            raise ValueError("transformer must be a TableTransformer object, a dictionary or None.")
        if not self._transformer.fit_complete:
            if self._transformer.needs_epsilon and (preprocessor_eps is None or preprocessor_eps == 0.0):
                raise ValueError("Transformer needs some epsilon to infer bounds.  If you know the bounds, pass them in to save budget.  Otherwise, set preprocessor_eps to a value > 0.0 and less than the training epsilon.  Preprocessing budget will be subtracted from training budget.")
            self._transformer.fit(
                data,
                epsilon=preprocessor_eps
            )
            eps_spent, _ = self._transformer.odometer.spent
            if eps_spent > 0.0:
                self.epsilon -= eps_spent
                print(f"Spent {eps_spent} epsilon on preprocessor, leaving {self.epsilon} for training")
                if self.epsilon < 10E-3:
                    raise ValueError("Epsilon remaining is too small!")
        train_data = self._transformer.transform(data)
        return train_data


    def fit(self,  
            key: KeyArray, 
            df: pd.DataFrame, 
            tolerance: float = 0.0, 
            adaptive_epoch=1,
            transformer=None,
            categorical_columns=[],
            ordinal_columns=[],
            continuous_columns=[],
            preprocessor_eps=0.0,
            nullable=False ):
        
        if type(df) is pd.DataFrame:
            self.original_column_names = df.columns
        
        categorical_check = (len(self._categorical_continuous(df)['categorical']) == len(list(df.columns)))
        if not categorical_check:
            raise ValueError('Please make sure that RAP gets categorical/ordinal\
                features only. If you are sure you only passed categorical, \
                increase the `thresh` parameter.')
        df = self._slide_range(df)

        train_data = self._get_train_data(
            df,
            style='cube',
            transformer=transformer,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns,
            continuous_columns=continuous_columns,
            nullable=nullable,
            preprocessor_eps=preprocessor_eps
        )
        print(train_data)

        if self._transformer is None:
            raise ValueError("We weren't able to fit a transformer to the data. Please check your data and try again.")

        cards = self._transformer.cardinality
        if any(c is None for c in cards):
            raise ValueError("The transformer appears to have some continuous columns. Please provide only categorical or ordinal.")

        colnames = ["col" + str(i) for i in range(self._transformer.output_width)]

        if len(cards) != len(colnames):
            raise ValueError("Cardinality and column names must be the same length.")

        domain = Domain(colnames, cards)
        print(colnames)
        data = pd.DataFrame(train_data, columns=colnames)
        data = Dataset(df=data, domain=domain)
        marginal_module2 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32])
        stat_module = ChainedStatistics([marginal_module2])
        stat_module.fit(data)
        print(stat_module)
        self.synthesizer = GSD(num_generations=self.num_generations,
                               domain=domain, 
                               data_size=self.data_size,
                               population_size_muta=self.population_size_muta,
                               population_size_cross=self.population_size_cross,
                               population_size=self.population_size,
                               muta_rate=self.muta_rate,
                               mate_rate=self.mate_rate,
                               print_progress=self.print_progress,
                               stop_early=self.stop_early,
                               stop_early_gen=self.stop_early_gen,
                               stop_eary_threshold=self.stop_early_threshold)
        self.res_df = self.synthesizer.fit(key, stat_module, data, tolerance, adaptive_epoch)

    def sample(self, n):
        assert n == self.data_size, "Can sample only the same amount as data_size provided during initialization"
        assert self.res_df is not None, "Please fit the synthesizer first."
        df = self._unslide_range(self.res_df)
        return df
