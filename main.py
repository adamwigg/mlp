"""ANN Experiment setup and execution"""
import ann
import report


def main():

    """Experiment settings"""
    config = {
        # Setup
        "data_file": "data/iris_str.csv",  # Relative filepath to csv file
        "data_seperator": ",",  # Usually ' ' or ','
        "header_rows": 0,  # Number of rows to ignore at start of file for header
        "random_seed": 2021,  # Used to initiate the numpy random generator
        # Dataset
        "data_split": (0.33, 0.33),  # Train, test - val is remaining if sum<99
        "target": -1,  # Target index, -1 being last column
        "normalize": True,  # normalize data
        "normalize_values": (-1, 1),  # Min and Max values
        # Parameters - multiple values in the list create a grid search
        "eta": [0.1],  # Training rate
        "epochs": [150],  # Number of epochs to train
    }

    # Directory/filename for save/load experiment (pickle file)
    experiment_dir = "experiments/"
    experiment_file = "test_01.pickle"

    """New experiment"""
    # Load config and data
    my_exp = ann.Experiment.new(config)
    # Create the model layers
    my_exp.nn = ann.NeuralNetwork(
        # Layers: name, input nodes, output nodes, activation
        ("input", 1, 4),
        ("hidden", 4, 8, ann.Sigmoid),
        ("hidden", 8, 8, ann.Sigmoid),
        ("output", 8, 4, ann.Softmax),
    )

    """Load experiment"""
    # my_exp = ann.Experiment.load(experiment_dir + experiment_file)

    """Run experiment"""

    my_exp.nn.fit()

    print(report.results(my_exp.results))

    my_exp.save(experiment_dir + experiment_file)

    print(f"\U0001F4BB Finished.")


if __name__ == "__main__":
    main()
