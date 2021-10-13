import ann
import report


""" Experiment settings """
config = {
    # Setup
    "data_file": "data/iris_str.csv",  # Relative filepath to csv file
    "data_seperator": ",",  # Usually ' ' or ','
    "header_rows": 0,  # Number of rows to ignore at start of file for header
    "random_seed": 2021,  # Used to initiate the numpy random generator
    # Dataset
    "train_test_split": (.33, .33,),  # Train > 0.65 will use '66/33' method
    "target": -1,  # Target index, -1 being last column
    "normalize": True,  # normalize data
    "normalize_values": (-1, 1),  # Min and Max values
    # Model settings
    "input_nodes": 12,  # Number of input values (features)
    "output_nodes": 3,  # Number of output values (classes)
    "hidden_layers": 2,  # Number of hidden layers
    "activator": ann.Softmax,  # alternatives: ann.Relu, ann.Sigmoid
    # Parameters - multiple values in the list create a grid search
    "eta": [0.1],  # Training rate
    "hidden_nodes": [8],  # Hidden nodes
    "max_epochs": [150],  # Number of epochs to train
}

# Directory/filename for save/load experiment (pickle file)
experiment_dir = "experiments/"
experiment_file = "test_01.pickle"

""" Create or load experiment"""
my_exp = ann.Experiment.new(config)
# my_exp = ann.Experiment.load(experiment_dir + experiment_file)

# print(my_exp)

""" Run the experiment"""
# my_exp.train()
# print(report.print_experiment(my_exp))

# my_exp.save(experiment_dir + experiment_file)
print('Done!')