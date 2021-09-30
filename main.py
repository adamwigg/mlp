"""
A Simple ANN - Soft Computing, Semester 2, 2021
"""
import ann

# Settings
config = {
    # Setup
    "data_file_path": 'data/my_data.csv',  # Relative filepath to csv or text file
    "data_seperator": ' ',  # Usually ' ' or ','
    "header_rows": 0,  # Number of rows to ignore at start of file
    "random_seed": 2021,  # Used to initiate the numpy random generator

    # Dataset
    "split": (0.3, 0.3),  # Eg (.33, .33) for 33-33-33 or (.66) for 66-33
    "target": -1,  # Target index, -1 being last column
    "normalize_values": (-1, 1),  # Min and Max values
    "normalize_range": [(0, 8)],  # List of (start, end) index ranges

    # Model input and output
    "input_nodes": 12,  # Number of input values (features)
    "output_nodes": 3,  # Number of output values (classes)
    "hidden_layers": (8, 2),  # hidden parameters (number of nodes, number of layers)

    # Hyperparameters - multiple list values create a grid search
    "eta": [0.1],  # Training rate
    "max_epochs": [150],  # Number of epochs to train
}


# Create experiment and prepare the data
my_exp = ann.Experiment.new(config)
my_exp.data_preperation()

# Alternatively load previous
# my_exp = ann.Experiment.load('file_path_to_pickle')

my_exp.train()

# acc, losss, w1, w2 = train(x, y, w1, w2, 0.1, 100)

# Report - 'summary' (default), 'details' or any 'Experiment' property 
my_exp.report()

# my_exp.save('file_path_to_pickle') # Optional
