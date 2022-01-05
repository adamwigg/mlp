"""
Metrics
-------
"""


def confusion_matrix(y_actual: np.ndarray, y_prediction: np.ndarray) -> pd.DataFrame:
    """Confusion matrix using the pandas crosstab - returns a dataframe"""
    df_confusion = pd.crosstab(
        pd.Series(y_actual),
        pd.Series(y_prediction),
        rownames=["Actual"],
        colnames=["Predicted"],
        margins=True,
    )
    return df_confusion


def macro_accuracy(y_actual: np.ndarray, y_prediction: np.ndarray) -> float:
    """Score the predictions"""
    tp = 0  # true positive
    fp = 0  # false positive
    tn = 0  # true negative
    fn = 0  # false negative
    cm = confusion_matrix(y_actual, y_prediction)
    diagonal = pd.Series(np.diag(cm), index=[cm.index, cm.columns]).tolist()
    for n in range(cm.shape[0]):
        tp += diagonal[n]
        tn += sum(diagonal) - diagonal[n]
        fn += sum(cm.iloc[n, :].tolist()) - diagonal[n]
        fp += sum(cm.iloc[:, n].tolist()) - diagonal[n]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy
