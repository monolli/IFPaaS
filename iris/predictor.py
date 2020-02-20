'''
This is a minimalistic Iris flower classifier predictor.
'''
from typing import List

import joblib
import numpy as np
from sklearn.datasets import load_iris


def predict_iris(data: List[float], model_path: str) -> List[str]:
    """Short summary.

    Parameters
    ----------
    data : List[float]
        The new data points that are going to be classifed.
        [['sepal length', 'sepal width', 'petal length', 'petal width'], ...]
    model_path : str
        The path where the trained model is stored.

    Returns
    -------
    List[str]
        A list containing the class of the provided objects.

    """
    # Load the dataset from the sklearn lib.
    iris = load_iris()
    # Load model
    clf = joblib.load(model_path)
    # Predict the class based on the data converted to numpy array
    Pred = clf.predict(np.array(data))
    # Return the labels
    return [iris.target_names[x] for x in Pred]
