"""
Single hidden layer feedforward multilayer perceptron
Adam Wigg (u913656)
University of Canberra, Soft Computing Unit, Semester 2, 2021
"""
import ann

# Settings
config = {
    # Setup
    'data_file_path': 'data/my_data.csv',
    'random_seed': 2021,

    # Dataset
    'split': (.3,.3),  # Eg (.33, .33) for 33-33-33 or (.66) for 66-33
    'target': -1,  # Target index
    'normalize_values': (-1, 1),  # Min and Max values
    'normalize_data_range': (0, 8), # Begining, end of range to normalize

    # Model settings
    'input_nodes': 12,  # Number of input values (features)
    'output_nodes': 3,  # Number of output values (classes)

    # parameters - multiple create grid search
    'hidden': [8],  # Hidden layer nodes
    'eta': [0.1],  # Training rate
    'epochs': [150]  # Number of epochs to train 
    }

# Create the model and prepare the data
nn = ann.Experiment.new(config)

# Alternatively load previous
# nn = mlp.Project.load('file_path_to_pickle') # Alternatively load previous (config above ignored)

nn.prepare_data()

# Run
nn.train()
nn.validate()
nn.test()

# Show the results
nn.report()

# nn.save('file_path_to_pickle') # Optional

