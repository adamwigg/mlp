"""
Activation Functions
--------------------
"""

from abc import ABC, abstractmethod
import numpy as np


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
