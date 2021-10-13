"""
Reports
"""


def experiment_data(experiment) -> str:
    """Print a summery description of an experiment"""
    report = f"Training - x: {experiment.x_train.shape} y: {experiment.y_train.shape}\nTesting - x: {experiment.x_test.shape} y: {experiment.y_test.shape} \nValidation - x: {experiment.x_val.shape} y: {experiment.y_val.shape}"
    return report


def best_result(results):
    return


def print_results(experiment):
    results = experiment.results
    try:
        report = ""
        for i, result in enumerate(results):
            pass

        best_result(results)

        return report
    except Exception:
        return "No results."


def print_confusion_matrix():
    pass


# def details(experiment):
#     """Report report of all the experiment/model details"""
#     report = f"===============================\n \
#         numInput = {experiment.numInput} \
#         numHidden = {experiment.numHidden} \
#         numOutput = {experiment.numOutput} \n\n\ \
#         inputs: \n {experiment.inputs}\n\n \
#         ihWeights: \n {experiment.ihWeights} \n\n \
#         hBiases: \n {experiment.hBiases} \
#         hOutputs: \n {experiment.hOutputs} \
#         hoWeights: \n {experiment.hoWeights}\n\n \
#         oBiases: \n {experiment.oBiases}\n\n \
#         hGrads: \n {experiment.hGrads}\n\n \
#         oGrads: \n {experiment.oGrads}\n\n \
#         ihPrevWeightsDelta: \n {experiment.ihPrevWeightsDelta}\n\n \
#         hPrevBiasesDelta: \n {experiment.hPrevBiasesDelta}\n\n \
#         hoPrevWeightsDelta: \n {experiment.hoPrevWeightsDelta}\n\n \
#         oPrevBiasesDelta: \n {experiment.oPrevBiasesDelta}\n\n \
#         outputs: \n {experiment.outputs}\n\n \
#         ===============================\n \
#         "
#     print(report)
