"""
Neural Network
--------------
"""


class Layer:
    """Basic general model layer"""

    def __init__(self, nodes) -> None:
        def rnd_array():
            return np.random.rand(1, nodes)

        # weights and biases
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
        # Create the model
        input_layer = Layer(_parent.no_features)
        self.model.append(input_layer)
        # Hidden layer/s
        for _ in range(hidden[0]):
            self.model.append(Layer(hidden[1]))
        self.activator = hidden[2]
        # Output layer
        self.model.append(Layer(len(_parent.labels)))
        self.output_activator = Softmax

    def forward(self) -> None:
        """Forward pass - call activation function with d=False"""
        pass

    def backward(self) -> None:
        """Back pass/propagation - call activation function with d=True for differention (gradient)"""
        pass

    def fit(self, eta) -> None:
        "Model training and validation"
        return "fit"
        # """Train the model"""

        #     for epoch in range(self._parent.max_epochs):
        #         start = datetime.time()

        #         finish = datetime.time()
        #         # feedback = f"Accuracy: {accuracy()} Time: {finish - start:.2f}"
        #         # progress_bar(epoch, epochs, feedback)

        # return # run result returned
