"""
Reports
-------
"""

from ann.experiment import Experiment, NeuralNetwork
from dataclasses import dataclass
import numpy as np


@dataclass()
class Result:
    """Dataclass for containing results of model runs"""

    nn: NeuralNetwork
    parameters: list
    val_actual: np.ndarray = None
    val_prediction: np.ndarray = None


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
