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
import itertools
import pickle
import datetime


"""
Utilities
---------
"""


def progress_bar(current: int, total: int, feedback: str = "", width=20) -> None:
    """Prints a progress bar given a current epoch, total epochs and feedback."""
    print("Running...")
    progress = "|" * int(current / total * width + 1)
    bar = "-" * (width - len(progress))
    print(f"\r[{progress}{bar}] Epoch: {current+1} of {total} -> {feedback}", end="\r")
    if current + 1 == total:
        print("\n")
        print(f"\U0001F600 Done!")


"""
Reports
-------
"""


def report_experiment(experiment: "Experiment") -> str:
    """Text summary of experiment parametres"""
    report_text = f"\
        File: {experiment.data_file} \n \
        Seed: {experiment.random_seed} \n \
        Hyperperameters: \n \
        - max epochs: {experiment.max_epochs} \n \
        - eta: {experiment.eta} \n \
        - hidden layers: {experiment.hidden} \n \
        "
    return report_text


def report_data_splits(experiment: "Experiment") -> str:
    """Text summary of experiment data"""
    report_text = f"\
        Training - x: {experiment.x_train.shape} \y: {experiment.y_train.shape}\n \
        Testing - x: {experiment.x_test.shape} y: {experiment.y_test.shape} \n \
        Validation - x: {experiment.x_val.shape} y: {experiment.y_val.shape} \
        "
    return report_text


def report_accuracy(scores: dict) -> str:
    """Text summary of metrics and confusion matrix"""
    report_text = ""
    for key, value in scores.items():
        if key != "confusion_matrix":
            report_text += f"{key}: {value:.2f}\n"
    report_text += f"Confusion Matrix:\n{scores['confusion_matrix'].to_string()}\n"
    return report_text


def report_result_list(experiment: "Experiment") -> str:
    report_text = ""
    for idx, result in enumerate(experiment.results):
        report_text += f"{idx:<2} - accuracy: {result.scores['accuracy']}, f1 score: {result.scores['f1_score']}"
    return report_text


"""
Data Functions
--------------
"""


def normalize_features(
    data: np.ndarray,
    target: int,
    norm: bool,
    norm_values: tuple[float, float],
) -> np.ndarray:
    """Normalize the feature data"""
    # separate features and targets
    targets = data[:, target]
    features = np.delete(data, target, 1)
    # normalize
    if norm == True:
        features_min = np.min(features)
        features_max = np.max(features)
        y = (features - (features_max + features_min) / 2) / (
            features_max - features_min
        )
        features = (
            y * (norm_values[1] - norm_values[0])
            + (norm_values[1] + norm_values[0]) / 2
        )
    # repack data - targets last
    norm_data = np.c_[features, targets]
    return norm_data


def split_xy(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split data into x (features) and y (targets)"""
    y = data[:, -1]
    x = np.delete(data, -1, 1)
    return x, y


def split_data(data: np.ndarray, split: tuple[float, float]) -> tuple[np.ndarray, ...]:
    """Create the train, test and validation datasets"""
    n_rows = data.shape[0]
    np.random.shuffle(data)
    if sum(split) >= 0.99:  # 66/33 method
        train_idx = int(round(n_rows * split[0]))
        train = data[:train_idx, :]
        test = data[train_idx:, :]
        val = train
    else:  # 33/33/33 method
        train_idx = int(round(n_rows * split[0]))
        test_idx = int(round(n_rows * split[1]) + train_idx)
        train = data[:train_idx, :]
        test = data[train_idx:test_idx, :]
        val = data[test_idx:, :]
    # x, y splits
    x_train, y_train = split_xy(train)
    x_test, y_test = split_xy(test)
    x_val, y_val = split_xy(val)

    return x_train, y_train, x_test, y_test, x_val, y_val


"""
Metrics
-------
"""


def confusion_matrix(y_actual: np.ndarray, y_prediction: np.ndarray) -> pd.DataFrame:
    """Confusion matrix using the pandas crosstab - returns a dataframe"""
    df_confusion = pd.crosstab(
        pd.Series(y_actual),
        pd.Series(y_prediction),
        rownames=["Actual"],
        colnames=["Predicted"],
        margins=True,
    )
    return df_confusion


def macro_accuracy(y_actual: np.ndarray, y_prediction: np.ndarray) -> float:
    """Score the predictions"""
    tp = 0  # true positive
    fp = 0  # false positive
    tn = 0  # true negative
    fn = 0  # false negative
    cm = confusion_matrix(y_actual, y_prediction)
    diagonal = pd.Series(np.diag(cm), index=[cm.index, cm.columns]).tolist()
    for n in range(cm.shape[0]):
        tp += diagonal[n]
        tn += sum(diagonal) - diagonal[n]
        fn += sum(cm.iloc[n, :].tolist()) - diagonal[n]
        fp += sum(cm.iloc[:, n].tolist()) - diagonal[n]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy



"""
Activation Functions
--------------------
"""


class Activator(ABC):
    """Activator interface (d = True for differentiation/backprop"""

    @abstractmethod
    def activation(self, x: np.ndarray, d: bool = False) -> np.ndarray:
        raise NotImplementedError("No activation method implemented.")


class Relu(Activator):
    """Compute rectified Linear Unit (relu) values for each sets of scores in x"""

    def activation(self, x: np.ndarray, d: bool = False) -> np.ndarray:
        if d:  # differentiation for backprop
            relu_x = np.heaviside(x, 1)
        else:
            relu_x = x
            relu_x[x < 0] = 0
        return relu_x


class Tanh(Activator):
    """Compute hyperbolic tan (tanh) values for each sets of scores in x"""

    def activation(self, x: np.ndarray, d: bool = False) -> np.ndarray:
        tanh_x = (2 / (1 + np.exp(-2 * x))) - 1
        if d:  # differentiation for backprop
            tanh_x = 1 - tanh_x ** 2
        return tanh_x


class Sigmoid(Activator):
    """Compute sigmoid values for each sets of scores in x"""

    def activation(self, x: np.ndarray, d: bool = False) -> np.ndarray:
        sigmoid_x = 1 / (1 + np.exp(-x))
        if d == True:
            sigmoid_x = sigmoid_x * (1 - sigmoid_x)
        return sigmoid_x


class Softmax(Activator):
    """Output function - Compute softmax values for each sets of scores in x"""

    def activation(self, x: np.ndarray, d: bool = False) -> np.ndarray:
        if d == True:
            raise NotImplementedError("Used for output layer only")
        sigmoid_x = np.exp(x)
        return sigmoid_x / sigmoid_x.sum()


"""
Network and Experiment
----------------------
"""


@dataclass()
class Result:
    """Dataclass for containing results of model runs"""

    nn: "NeuralNetwork"
    parameters: list
    val_actual: np.ndarray = None
    val_prediction: np.ndarray = None

    @property
    def scores(self):
        pass

    def report_scores(self) -> str:
        return report_result_scores(self.scores)


class Layer:
    """Basic general model layer"""

    def __init__(self, nodes) -> None:
        # weights and biases
        def rnd_array():
            return np.random.rand(1, nodes)

        self.w: np.ndarray = rnd_array()
        self.b: np.ndarray = rnd_array()
        self.z: np.ndarray = rnd_array()
        # derivatives
        self.da: np.ndarray = rnd_array()
        self.dw: np.ndarray = rnd_array()
        self.db: np.ndarray = rnd_array()
        self.dz: np.ndarray = rnd_array()


class NeuralNetwork:
    """Main ANN class with forward pass and back propagation functions"""

    def __init__(
        self, _parent: "Experiment", hidden: tuple[int, int, Activator]
    ) -> None:
        self._parent: Experiment = _parent
        self.model: list[Layer] = []
        # Build model
        input_nodes = _parent.no_features
        output_nodes = len(_parent.labels)
        # Create the model
        # Input layer
        input_layer = Layer(_parent.no_features)
        self.model.append(input_layer)
        # Hidden layer/s
        for _ in range(hidden[0]):
            self.model.append(Layer(hidden[1]))
        self.activator = hidden[2]
        # Output layer
        output_layer = Layer(
            "output", np.random.rand(hidden[1], output_nodes, Softmax())
        )
        self.model.append(output_layer)

    def forward(self) -> None:
        """Forward pass - call activation function with d=False"""
        pass

    def backward(self) -> None:
        """Back pass/propagation - call activation function with d=True for differention (gradient)"""
        pass

    def fit(self, eta) -> None:
        "Model training and validation"
        pass
        # """Train the model"""

        #     for epoch in range(self._parent.max_epochs):
        #         start = datetime.time()

        #         finish = datetime.time()
        #         # feedback = f"Accuracy: {accuracy()} Time: {finish - start:.2f}"
        #         # progress_bar(epoch, epochs, feedback)

        # return # run result returned

    def test(self) -> None:
        """Model testing run"""
        #     print(report.print_results(self))
        pass


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
        """Run the ann experiment"""
        parameters = itertools.product(*[self.eta, self.hidden_layers])
        for param in parameters:
            eta, hidden = param
            nn = NeuralNetwork(self, hidden)
            print(f"Fitting {hidden} with eta of {eta}")
            result = nn.fit(eta)
            self.results.append(nn.fit(eta))
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
