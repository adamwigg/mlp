"""
Collection of reports and plots
"""
# General reports
def details(experiment):
    """Report report of all the experiment/model details"""
    report = f"===============================\n \
        numInput = {experiment.numInput} \
        numHidden = {experiment.numHidden} \
        numOutput = {experiment.numOutput} \n\n\ \
        inputs: \n {experiment.inputs}\n\n \
        ihWeights: \n {experiment.ihWeights} \n\n \
        hBiases: \n {experiment.hBiases} \
        hOutputs: \n {experiment.hOutputs} \
        hoWeights: \n {experiment.hoWeights}\n\n \
        oBiases: \n {experiment.oBiases}\n\n \
        hGrads: \n {experiment.hGrads}\n\n \
        oGrads: \n {experiment.oGrads}\n\n \
        ihPrevWeightsDelta: \n {experiment.ihPrevWeightsDelta}\n\n \
        hPrevBiasesDelta: \n {experiment.hPrevBiasesDelta}\n\n \
        hoPrevWeightsDelta: \n {experiment.hoPrevWeightsDelta}\n\n \
        oPrevBiasesDelta: \n {experiment.oPrevBiasesDelta}\n\n \
        outputs: \n {experiment.outputs}\n\n \
        ===============================\n \
        "
    print(report)

def attr(experiment, attr):
    """Print an attribute of the experiment - useful for debugging"""
    print (getattr(experiment, attr))


def main():
    pass

if __name__ == '__main__':
    main()