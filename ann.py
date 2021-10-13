"""Simple artificial neural network (ann)"""

import numpy as np
import pandas as pd
import report

# Python 3.9+ Standard library
from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import itertools
import math
import pickle
import datetime


"""Utilities"""


def progress_bar(current: int, total: int, feedback: str = "", width=20) -> None:
    """
    Prints a progress bar given a current epoch, total epochs and feedback.
    """
    progress = "|" * int(current / total * width + 1)
    bar = "-" * (width - len(progress))
    print(f"\r[{progress}{bar}] Epoch: {current+1} of {total} -> {feedback}", end="\r")
    if current + 1 == total:
        print("\n")
        print(f"\U0001F600 Done!")


# Data and Results


def normalize_features(
    data: Any,
    target: int,
    norm: bool,
    norm_values: tuple[float, float],
) -> Any:
    """Normalize the feature data"""
    features = np.delete(data, target, 1)
    targets = data[:, target]
    if norm == True:
        domain = np.min(features), np.max(features)
        y = (features - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        features = (
            y * (norm_values[1] - norm_values[0])
            + (norm_values[1] + norm_values[0]) / 2
        )
    norm_data = np.c_[features, targets]
    return norm_data


def split_xy(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split data into x (features) and y (targets)"""
    y = data[:, -1]
    x = np.delete(data, -1, 1)
    return x, y


def split_data(data: np.ndarray, split: tuple[float, float]):
    """Create the train, test and validation datasets"""
    n_rows = data.shape[0]
    np.random.shuffle(data)
    if sum(split) >= 0.99:
        train_idx = [round(n_rows * split[0])]
        train = data[:train_idx, :]
        test = data[train_idx:, :]
        val = train
    else:
        train_idx = round(n_rows * split[0])
        test_idx = round(n_rows * split[1]) + train_idx
        train = data[:train_idx, :]
        test = data[train_idx:test_idx, :]
        val = data[test_idx:, :]

    x_train, y_train = split_xy(train)
    x_test, y_test = split_xy(test)
    x_val, y_val = split_xy(val)

    return x_train, y_train, x_test, y_test, x_val, y_val


# Metrics


def confusion_matrix():
    pass


# def score():
#     TP = 100 - FN - FP - TF
#     #print(FN, FP, TP, FN+FP+TP+TF)
#     precision = TP / (TP + FP)
#     accuracy = (TP + TN)/(TP + TN + FP + FN)
#     recall = TP / (TP + FN)
#     f1_score = 2 * precision * recall / (precision + recall)
#     pass


# Activation Functions


class Activator(ABC):  # Activator interface
    @abstractmethod
    def activation_function(self, x: float, d: bool = False) -> float:
        raise NotImplementedError


class Relu(Activator):
    """Compute relu values for each sets of scores in x"""

    def activation_function(self, x: float, d: bool = False) -> float:
        relu_x = x
        relu_x[x < 0] = 0
        return relu_x


class Sigmoid(Activator):
    """Compute sigmoid values for each sets of scores in x"""

    def activation_function(self, x: float, d: bool = False) -> float:
        sigmoid_x = 1 / (1 + math.exp(-x))
        return sigmoid_x


class Softmax(Activator):
    """Compute softmax values for each sets of scores in x"""

    def activation_function(self, x: float, d: bool = False) -> float:
        e_x = np.exp(x - np.max(x))
        softmax_x = e_x / e_x.sum(axis=0)
        return softmax_x


# Network and Experiment


class NeuralNetwork:
    """Main ANN class with forward pass and back propagation functions"""

    def __init__(*args) -> None:
        pass

    def forward(self, x_train):
        pass

    def backward():
        pass

    def fit(self) -> None:
        """Train the model"""
        for parameters in self.gridsearch():
            eta, hidden, epochs = parameters  # unpack parameters
            # Initialise model
            self.model = NN(
                self.input_nodes,
                self.hidden_layers,
                hidden,
                self.output_nodes,
                self.activator,
            )
            # Training loop
            for epoch in range(epochs):
                start = datetime.time()

                finish = datetime.time()
                # feedback = f"Accuracy: {accuracy()} Time: {finish - start:.2f}"
                # progress_bar(epoch, epochs, feedback)

            print(report.print_results(self))

    def forward(self, x_train):
        pass

    def back_prop(self):
        pass


class Experiment:
    """Contains all the settings, data and results of an experiment"""

    def __init__(self, config: dict) -> None:
        # Data file
        self.data_file: str = None
        self.data_seperator: str = ","
        self.header_rows: int = 0
        # Processing
        self.random_seed: int = 1968
        self.data_split: tuple[float, float] = (0.33, 0.33)
        self.target: int = -1
        self.normalize: bool = True
        self.normalize_values: tuple[float, float] = (-1, 1)
        # Neural Network
        self.nn: NeuralNetwork = None
        # Model and Parametres
        self.eta: list[float] = [0.1]
        self.epochs: list[int] = [150]
        # Experiment data
        self.labels: list = []
        self.x_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.x_test: np.ndarray = None
        self.y_test: np.ndarray = None
        self.x_val: np.ndarray = None
        self.y_val: np.ndarray = None
        # Output
        self.results: list[Result] = None
        # Load config values
        for key, value in config.items():
            setattr(self, key, value)
        # Grids of parameters as a generator
        self.parameters = (
            p
            for p in list(
                itertools.product(*[self.eta, self.hidden_nodes, self.max_epochs])
            )
        )

    @classmethod
    def new(cls, config: dict) -> Any:  # Main constructor
        """Constructor for new project from csv file"""
        new_exp = cls(config)
        print("New experiment...\n")
        np.random.seed(new_exp.random_seed)  # Initialize numpy random seed
        # Load data and process lables (classes)
        try:
            df = pd.read_csv("data/iris_str.csv", header=0, delimiter=",")
            print(f"\U00002714 CSV file loaded")
            new_exp.labels = df.iloc[:, -1].unique().tolist()
            df.iloc[:, -1] = df.iloc[:, -1].apply(lambda c: new_exp.labels.index(c))
            new_data = df.to_numpy()
            print(f"\U00002714 Labels extracted")
        except Exception as e:
            print(f"\U0001F4A5 New Experiment: Data file not found or unreadable.\n")
            print(e)
        # Process data and create data loader (splits)
        try:
            norm_data = normalize_features(
                new_data, new_exp.target, new_exp.normalize, new_exp.normalize_values
            )
            print(f"\U00002714 Features normalized")
        except Exception as e:
            f"\U0001F4A5 New Experiment: Data loaded but failed processing - check settings.\n {e}"
        (
            new_exp.x_train,
            new_exp.y_train,
            new_exp.x_test,
            new_exp.y_test,
            new_exp.x_val,
            new_exp.y_val,
        ) = split_data(norm_data, new_exp.data_split)
        print(f"\U00002714 Data prepared for training\n")
        print(report.data_splits(new_exp))
        print(f"\nTarget labels: {new_exp.labels}\n")
        print(f"\U0001F600 Experiment created successfully.\n")

        return new_exp

    @staticmethod
    def load(filepath: str) -> Any:  # Load previous
        """Load project from binary"""
        print("Loading experiment...\n")
        try:
            with open(filepath, "rb") as f:
                loaded_exp = pickle.load(f)
                print(report.experiment_data(loaded_exp))
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


def main():
    pass


if __name__ == "__main__":
    main()
