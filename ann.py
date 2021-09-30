# Imports
import numpy as np
import matplotlib as plt
# Standard library imports
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Protocol
import pickle


"""
Utilities
"""

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
 
"""
Activation Functions
"""
class Activator(Protocol):
    @abstractmethod
    def activation_function(self, x: float) -> float:
        raise NotImplementedError


class NoActivator:
    def activation_function(self, x: float) -> float:
        return x

class ReluActivator:
    def activation_function(self, x: float) -> float:
        relu_x = x
        relu_x[x < 0] = 0
        return relu_x

class SoftmaxActivator:
    """Compute softmax values for each sets of scores in x."""
    def activation_function(self, x: float) -> float:
        e_x = np.exp(x - np.max(x))
        soft_x = e_x / e_x.sum(axis=0)
        return soft_x


"""
Layers
"""
    
class Layer:
    def __init__(self, layer_type: str,  prev_nodes: int, layer_nodes: int, activation: Activator) -> None:
        self.type = layer_type
        self.w = np.random.uniform(low=-0.1, high=0.1, size=(prev_nodes, layer_nodes))
        self.activation = activation 
    
    def __repr__(self) -> str:
        return (f'{self.type} layer (weights: {self.w.shape})')


def forward_pass():
    pass

def back_propagation():
    pass

"""
Dataset
"""

def one_hot(targets: Any) -> tuple[Any, Any]: 
    """Simple one hot encoder for target values"""
    unique, inverse = np.unique(targets, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot, unique


def normalize(data: Any, ranges: list, values: tuple = (-1, 1)) -> Any:
    """Takes in a numpy array and normalizes the the 'ranges' for the 'values and returns the array'"""
    min, max = values
    values = data

class Dataset:
    """
    Contains a set of data and:
    - test, train, validation sets.
    - preprocessing functions
    Created by the AnnProject class when starting 
    a new project or loading from file.
    """
    def __init__(self, data, normalize, target_idx) -> None:
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


class NN:    
    def __init__(self, input_nodes, hidden_nodes, hidden_layers, output_nodes) -> None:
        self.model = []
        self.model.append(Layer('input', input_nodes, hidden_nodes, SoftmaxActivator))
        if len(hidden_layers) > 1:
            for layer in hidden_layers[1:]:
                self.model.append(Layer('hidden', hidden_nodes, hidden_nodes, SoftmaxActivator))
        self.output_layer = Layer('optput', hidden_nodes, output_nodes, NoActivator)

        for layer in self.model:
            print(layer)


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
        self.config: Dict = config
        self.dataset: Dataset = None
        self.hyper_params: list = []
        self.total_epochs: int = 100
        self.results: dict = None
        self.model: NN = None

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

    def gridsearch(self) -> None:
        """Creates a list of hyper perametre parametres as an input to a training run"""
        # hyper_params = []
        # for eta in hyper_parametres['eta']:
        #     for hidden in hyper_parametres['hidden']:
        #         hyper_params.append((eta, hidden))
        # self.hyper_params = hyper_params
        return []
    
    def train(self) -> None:
        """Training run loop"""
        training_parameters = self.gridsearch()
        for params in self.hyper_params:
            eta, hidden = params  # unpack parameters 
            self.model = NN(self.config['input_nodes'], self.config['hidden_layers'][0], self.config['hidden_layers'][1], self.config['output_nodes']) 
            for epoch in range(self.total_epochs):
                

                accuracy = f'Accuracy: {calc_accuracy()}'
                progress_bar(epoch, self.total_epochs, 'Epoch', accuracy)

    def report(self, details: str = 'summary') -> None:
        pass # print(make_report(self, details))
        

def main():
    pass

if __name__ == '__main__':
    main()

"""
References:

"""