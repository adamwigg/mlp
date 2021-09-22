"""
A Simple ANN - Semester 2, 2021
"""
import ann

# Settings
experiment_config = {
    # Setup
    "data_file_path": 'data/my_data.csv',  # Relative filepath to csv or text file
    "data_seperator": ' ',  # Data seporator usually ' ' or ','
    "header_rows": 0,  # Number of rows to ignore at start of file
    "random_seed": 2021,  # Used to initiate the numpy random generator
    # Dataset
    "split": (0.3, 0.3),  # Eg (.33, .33) for 33-33-33 or (.66) for 66-33
    "target": -1,  # Target index
    "normalize_values": (-1, 1),  # Min and Max values for normalization
    "normalize_range": [(0, 8)],  # List of (start, end) index ranges to normalize
    # Model input and output
    "input_nodes": 12,  # Number of input values (features)
    "output_nodes": 3,  # Number of output values (classes)
    "hidden_layers": 1
}

hyper_parameters = {
    # parameters - multiple list values create a grid search
    "eta": [0.1],  # Training rate
    "hidden": [8],  # Hidden layer nodes
    # Number of hidden layers
    "max_epochs": 150,  # Number of epochs to train
}

# Create experiment and prepare the data
nn = ann.Experiment.new(experiment_config)
nn.data_preperation()

# Alternatively load previous
# nn = mlp.Experiment.load('file_path_to_pickle') # Alternatively load previous experiment

nn.gridsearch(hyper_parameters)
nn.train()

# Report defaults to 'summary' other options:
# 'details', 'confusion_matrix'
# or any Experiment property eg 'config' or 'dataset'
nn.report()

# nn.save('file_path_to_pickle') # Optional
