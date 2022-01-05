"""
Simple ANN (ann.py) Experiment setup and execution
Adam Wigg @ University of Canberra for Soft Computing S2/2021
"""
import ann

def main():
    config = {  # All new experiment variables
        # Data
        "data_file": "data/iris_str.csv",  # Relative filepath to csv file
        "data_seperator": ",",  # Usually ' ' or ','
        "header_rows": 0,  # Number of rows to ignore
        "normalize": True,  # normalize data
        "normalize_values": (-1, 1),  # Min and Max values
        "target_index": -1,  # Target index, -1 being last column
        "data_split": (0.66, 0.33),  # Train, test (val = test if sum >= .99 else remaining)
        # Misc
        "random_seed": 2021,  # Used to initiate the random generator
        # Experiment parameters
        "max_epochs": 200,  # Max number of epochs to train
        # eta and hidden - multiple values in the list create a grid search
        "eta": 0.1,
        # Hidden layer in the form (no of hidden layers, no of nodes, activation function)
        # Available activation functions: ann.Relu(), ann.Tanh(), ann.Sigmoid() (output uses softmax)
        "hidden_layers": (2, 8, ann.Relu())
    }

    """
    Experiment file
    - Directory/filename for save experiment (pickle file)
    """
    experiment_dir = "experiments/"
    experiment_file = "exp01_02_08_sigmoid.pickle"

    """
    Run experiment
    - New experiment - loads csv data and uses config to build model
    """
    my_experiment = ann.Experiment.new(config)

    """
    Alternatively, load experiment
    - Loads all the previous run properties (including data)
    """
    # my_experiment = sann.Experiment.load(experiment_dir + experiment_file)

    """Modifying loaded experiment examples"""
    # my_experiment.max_epochs = 500
    # my_experiment.eta = 0.1
    # my_experiment.hidden_layers = (4, 12, ann.Relu())

    """Run the experiment and report the results"""
    my_experiment.run()
    

    """Save the experiment"""
    my_experiment.save(experiment_dir + experiment_file)

   
    print(f"\U0001F4BB Done!")

if __name__ == "__main__":
    main()
