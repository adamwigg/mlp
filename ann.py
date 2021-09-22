# Imports
import numpy as np
import pandas as pd
import matplotlib as plt
# Standard library imports
from typing import Protocol, Any
import pickle


def progress_bar(current: int, total: int, title: str = 'Progress', data: str = '', width = 20) -> None:
    """
    Prints a progress bar given a current step and total.
    Optional 'title' and 'data' values.
    """
    progress = '|' * int(current / total * width + 1)
    bar  = '-' * (width - len(progress))
    show = ''
    if data:
        show = f'-> {data}'
    print(f'\r[{progress}{bar}] {title.capitalize()}: {current+1}/{total} {show}', end='\r')
    if current + 1 == total:
        print('\n')
        print(f"\N{grinning face} Done!")

def make_report(experiment: 'Experiment', scope: str = 'summary') -> str:
    """Return a string that summarises the experiment or reports on a attribute of an experiment"""
    if scope == 'summary':
        report = 'Summary of experiment'
    elif scope == 'details':
        report = f"===============================\n \
            numInput = {experiment.numInput} \
            numHidden = {experiment.numHidden} \
            numOutput = {experiment.numOutput} \n\n\ \
            inputs: \n {experiment.inputs}\n\n \
            ihWeights: \n {experiment.ihWeights} \n\n \
            hBiases: \n {experiment.hBiases} \
            hOutputs: \n {experiment.hOutputs} \
            hoWeights: \n {experiment.hoWeights}\n\n \
            oBiases: \n {experiment.oBiases}\n\n \
            hGrads: \n {experiment.hGrads}\n\n \
            oGrads: \n {experiment.oGrads}\n\n \
            ihPrevWeightsDelta: \n {experiment.ihPrevWeightsDelta}\n\n \
            hPrevBiasesDelta: \n {experiment.hPrevBiasesDelta}\n\n \
            hoPrevWeightsDelta: \n {experiment.hoPrevWeightsDelta}\n\n \
            oPrevBiasesDelta: \n {experiment.oPrevBiasesDelta}\n\n \
            outputs: \n {experiment.outputs}\n\n \
            ===============================\n \
            "
        print(report)
    else:
        try:
            attribute_to_print = getattr(experiment, scope)
            report = attribute_to_print
        except Exception as e:
            print(f"\N{frowning face} Attribute not found or some other stuff up...")
            print(e)

    return report
 

def preprocess(data: Any, ranges: list, values: tuple = (-1, 1)) -> Any:
    """Takes in a numpy array and normalizes the the 'ranges' for the 'values and returns the array'"""
    min, max = values
    values = data


def weights_update(weights: Any, eta: float):
    new = weights - eta * weights
    return new 

# Activation Functions
class Activator(Protocol):
    def activation_function(self) -> None:
        pass

class NoActivator:
    def activation_function(self, x):
        return x

class ReluActivator:
    def activation_function(self, x):
        relu_x = x
        relu_x[x < 0] = 0
        return relu_x

class SoftmaxActivator:
    """Compute softmax values for each sets of scores in x. https://stackoverflow.com/a/38250088 """
    def activation_function(self, x):
        e_x = np.exp(x - np.max(x))
        soft_x = e_x / e_x.sum(axis=0)
        return soft_x


class Layer:
    def __init__(self, prev_nodes: int, layer_nodes: int, obj: Activator) -> None:
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(prev_nodes, layer_nodes))
        self.activator = obj


class Dataset:
    """
    Contains a set of data and:
    - test, train, validation sets.
    - preprocessing functions
    Created by the AnnProject class when starting 
    a new project or loading from file.
    """
    def __init__(self, data) -> None:
        self.data = data
        self.normalize_parameters: list = []
        self.target_column_index: int = -1 

    @property
    def rows(self) -> int:
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        return self.data.shape[1]

    def __repr__(self) -> str:
        """Pretty string of an array"""
        return np.array2string(self.data, formatter={'float_kind':lambda x: "%.4f" % x})

class Model:    
    def __init__(self, input_nodes, hidden_nodes, hidden_layers, output_nodes) -> None:
        self.model = []
        self.model.append(Layer(input_nodes, hidden_nodes, SoftmaxActivator))
        if hidden_layers > 1:
            for layer in range(hidden_layers-1):
                self.model.append(Layer(hidden_nodes, hidden_nodes, SoftmaxActivator))
        self.output_layer = Layer(hidden_nodes, output_nodes, NoActivator)

    def initialize() -> None:  # Populate weights and biases with random numbers
        pass


class Experiment:
    """
    An experiment contains:
    the dataset, neural network, settings, and last run results.
    A new project can be:
    - started from scratch with a csv using 'from_csv'.
    - saved to a binary using 'project_save'
    - restored (loaded) from binary using 'project_load' 
    """
    def __init__(self, config) -> None:
        self.config = config
        self.dataset: Dataset = None
        self.hyper_params: list = []
        self.total_epochs: int = 100
        self.results: dict = None
        self.model: Model = None

        np.random.seed(config['random_seed']) # Seed numpy generator

    @classmethod
    def new(cls, config: dict) -> 'Experiment':
        """Constructor for new project from csv file"""
        try:
            csv_data = np.genfromtxt(config['data_file_path'], 
                                    delimiter=config['data_seperator'],
                                    skip_header=config['header_rows'])
            new_experiment = cls(config)
            new_data = preprocess(csv_data, config['normalize_range'], config['normalize_values'])
            new_experiment.dataset = Dataset(new_data, config['target'], config['split'])
            print(f"\N{grinning face} {filepath} -> Created.")
            return new_experiment
        except Exception as e:
            print(f"\N{frowning face} Data file not found, bad config or other sad thing...")
            print(e)

    @staticmethod
    def load(filepath: str) -> 'Experiment':
        """Load project from binary"""
        try:
            with open(filepath, 'rb') as f:
                loaded_experiment =  pickle.load(f)
                print(loaded_experiment.config)
                print(f"\N{grinning face} {filepath} -> Loaded.")
                return loaded_experiment
        except Exception as e:
            print(f"\N{frowning face} File not found, bad config or other nastyness...")
            print(e)
        
    def save(self, filepath: str) -> None:
        """Save project to a reloadable binary"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
                print(f"\N{grinning face} {filepath} -> Saved.")
        except Exception as e:
            print(f"\N{frowning face} File not found or bad stuff happened...")
            print(e)

    def gridsearch(self, hyper_parametres: dict) -> None:
        """Creates a list of hyper perametre parametres as an input to a training run"""
        hyper_params = []
        for eta in hyper_parametres['eta']:
            for hidden in hyper_parametres['hidden']:
                hyper_params.append((eta, hidden))
        self.hyper_params = hyper_params
    
    def train(self) -> None:
        """Training run loop"""
        for params in self.hyper_params:
            eta, hidden = params  # unpack parameters 
            self.model = Model(self.config['input_nodes'], hidden, self.config['hidden_layers'], self.config['output_nodes']) 
            for epoch in range(self.total_epochs):
                

                accuracy = f'Accuracy: {calc_accuracy()}'
                progress_bar(epoch, self.total_epochs, 'Epoch', accuracy)

    def report(self, details: str = 'summary') -> None:
        print(make_report(self, details))
        

def main():
    pass

if __name__ == '__main__':
    main()
