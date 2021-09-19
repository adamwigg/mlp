# Imports
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib as plt
# Standard library imports (Python 3.4+)
from os import name
from typing import List
from dataclasses import dataclass
from pprint import pprint
import pickle
import random


class Layer:
    pass


class Activation:
    pass


class Normalizer:
    pass


@dataclass
class Dataset:
    """
    Contains a set of data and:
    - test, train, validation sets.
    - preprocessing functions
    Created by the AnnProject class when starting 
    a new project or loading from file.
    """
    data: np.Array = None
    normalize_parameters: List = []
    target_column_index: int = -1  # Defaults to the last column

    @property
    def rows(self) -> int:
        return self.data.shape[0]
    
    @property
    def cols(self) -> int:
        return self.data.shape[1]

    def __repr__(self) -> str:
        """Pretty string of an array"""
        return np.array2string(self.data, formatter={'float_kind':lambda x: "%.4f" % x})


@dataclass
class NeuralNetwork:
    pass


class Experiment:
    """
    An ANN project that contains:
    the dataset, neural network, settings, and last run results.
    A new project can be:
    - started from scratch with a csv using 'from_csv'.
    - saved to a binary using 'project_save'
    - restored (loaded) from binary using 'project_load' 
    """
    def __init__(self, data, config):
        self.config = config
        self.raw_data = data
        self.confusion_matrix = None
        self.report_data = None
        self.dataset = Dataset
        self.nn = NeuralNetwork

    @classmethod
    def new(cls, config):
        """Constructor for new project from csv file"""
        filepath = config['data_directory'] + config['data_file_name']
        df = pd.read_csv(filepath)
        return cls(df.values, config)

    @staticmethod
    def load(filepath):
        """Load project from binary"""
        with open(filepath, 'rb') as f:
            proj =  pickle.load(f)
        try:
            pprint(proj.config)
            print(f'{filepath} Loaded.')
        except ValueError as e:
            print('Config not found.')
            print(e)
        return proj

    def save(self,filepath):
        """Save project to a reloadable binary"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def report():

##########################

class neural_net:
    def __init__(self, experiment) -> None:
        self.experiment = experiment
        # Model structure
        hidden = self.experiment.config['hidden']
        self.hidden_layer = ann.layer(input_nodes, hidden) 
        self.output = ann.layer(hidden, output_nodes) 
    
    def prepare_data(self):
        """Dataset propcessing"""
        # Normalize - can be repeated 
  

    def forward(self, x):
        x = self.layer(x)
        x = ann.softmax(x)
        x = self.output(x)
        return x

    def train(self):
        """Training loop"""
        for epoch in range(nn.config['epochs']):
            pass

    def validate(self):
        pass
    def test(self):
        pass

    def report(self):
        pass




def main():
    pass

if __name__ == 'main':
    main()