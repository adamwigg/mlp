"""
A simple Artificial Neural Network (ANN)
Adam Wigg @ University of Canberra for Soft Computing S2/2021
"""

# Installed libraries
import numpy as np
import pandas as pd

# Python 3.9+ Standard libraries
from abc import ABC, abstractmethod
from dataclasses import dataclass
from operator import attrgetter
import pickle
import datetime


class Experiment:
    """Contains all the settings, data and results of an experiment"""

    def __init__(self) -> None:
        # Data file
        self.data_file: str = ""
        self.data_seperator: str = ","
        self.header_rows: int = 0
        # Data processing
        self.random_seed: int = 1968
        self.data_split: tuple[float, float] = (0.33, 0.33)
        self.target_index: int = -1
        self.normalize: bool = False
        self.normalize_values: tuple[float, float] = (-1, 1)
        #  Model and Parametres
        self.max_epochs: list[int] = 100
        self.eta: list[float] = [0.1]
        self.hidden_layers: list[tuple[int, int, Activator]] = [(1, 8, Relu())]
        # Data sets
        self.labels: list = []
        self.no_features: int = 1
        self.x_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.x_test: np.ndarray = None
        self.y_test: np.ndarray = None
        self.x_val: np.ndarray = None
        self.y_val: np.ndarray = None
        # Output
        self.results: list[Result] = []

    @classmethod
    def new(cls, config: dict) -> "Experiment":  # Main constructor
        """Constructor for new project from csv file"""
        new_exp = cls()
        # Load config values
        for key, value in config.items():
            setattr(new_exp, key, value)
        print("New experiment...\n")
        # Initialize numpy random seed
        np.random.seed(new_exp.random_seed)
        # Load data and process lables (classes)
        try:
            # read file
            df = pd.read_csv(
                "data/iris_str.csv",
                header=new_exp.header_rows,
                sep=new_exp.data_seperator,
            )
            print(f"\U00002714 CSV file loaded")
            # extract target labels and create index references - pandas (common seporators should be sniffed)
            new_exp.labels = df.iloc[:, new_exp.target_index].unique().tolist()
            df.iloc[:, new_exp.target_index] = df.iloc[:, new_exp.target_index].apply(
                lambda idx: new_exp.labels.index(idx)
            )
            # convert to np array
            new_data = df.to_numpy()
            new_exp.no_features = new_data.shape[0] - 1
            print(f"\U00002714 Labels extracted")
        except Exception as e:
            print(
                f"\U0001F4A5 New Experiment: Data file not found or unreadable.\n {e}"
            )
        # Process data and create data loaders
        try:
            norm_data = normalize_features(
                new_data,
                new_exp.target_index,
                new_exp.normalize,
                new_exp.normalize_values,
            )
            print(f"\U00002714 Features normalized")
        except Exception as e:
            print(
                f"\U0001F4A5 New Experiment: Data loaded but failed normalization - check settings.\n {e}"
            )
        # Data splits
        (
            new_exp.x_train,
            new_exp.y_train,
            new_exp.x_test,
            new_exp.y_test,
            new_exp.x_val,
            new_exp.y_val,
        ) = split_data(norm_data, new_exp.data_split)
        print(f"\U00002714 Data prepared for training\n")
        print(report_data_splits(new_exp))
        print(f"\nTarget labels: {new_exp.labels}\n")
        print(f"\U0001F600 Experiment created successfully.\n")
        return new_exp

    @staticmethod
    def load(filepath: str) -> None:  # Load previous
        """Load project from binary (pickle)"""
        print("Loading experiment...\n")
        try:
            with open(filepath, "rb") as f:
                loaded_exp = pickle.load(f)
                print(report_data_splits(loaded_exp))
                print(f"Target labels: {loaded_exp.labels}\n")
                np.random.seed(loaded_exp.random_seed)  # Initialize numpy random seed
                print(f"\U0001F600 Experiment loaded.\n")
                return loaded_exp
        except Exception as e:
            print(f"\U0001F4A5 Load Experiment: File not loaded - {e}")

    def save(self, filepath: str) -> None:  # Pickle Experiment instance
        """Save project to a reloadable binary"""
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
                print(f"\U0001F600 Experiment saved.\n")
        except Exception as e:
            print(f"\U0001F4A5 File not saved\n")
            print(e)

    def run(self) -> None:
        """Run ann experiment"""
        nn = NeuralNetwork(self)

        print(
            f"Fitting: \n \
                {self.hidden_layers[0]} hidden layers of {self.hidden_layers[1]} nodes \n \
                with eta of {self.eta}"
        )
        result = nn.fit(self.eta)
        self.results.append(nn.fit(self.eta))
        nn.test()
        self.result()

    def result(self, result_idx: int = -1):
        """Print a result given the index in the results list"""
        try:
            print(report_result_scores(self.results[result_idx].scores))
        except IndexError:
            print("No result found.")

    def all_results(self):
        """Print a the contents of the results list"""
        try:
            print(report_result_list(self))
        except IndexError:
            print("No result found.")

    def best_result(self, metric: str = "accuracy"):
        """Print the best result by the given metric from the results list"""
        best = max(self.results, key=attrgetter(score[metric]))
        print(report_result_scores(best.scores))


def main():
    pass


if __name__ == "__main__":
    main()
