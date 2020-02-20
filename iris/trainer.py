'''
This is a minimalistic Iris flower classifier trainer.
'''
import os
from typing import List

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def data_spliter(X: np.ndarray, y: np.ndarray, seed: int = 997) -> List[np.ndarray]:
    """Split the data using a seed for reproducibility reasons.

    Parameters
    ----------
    X : np.ndarray
        Data containing the features.
    y : np.ndarray
        Data containing the targets.
    seed : int
        A number that is going to be used to set the random state.

    Returns
    -------
    List[np.ndarray]
        A list with the X and y split in train and test.
        [X_train, X_test, y_train, y_test]

    """
    return train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True)


def main() -> None:
    """Executes the program.

    Returns
    -------
    None
        Description of returned object.

    """
    # Load the dataset from the sklearn lib.
    iris = load_iris()
    # Break it in two separate dataset, data and target.
    X = iris.data
    y = iris.target
    # Split the dataset in two, training and test.
    # The classes are balanced and with a reasonable population, so it is ok to shuffle
    X_train, X_test, y_train, y_test = data_spliter(X, y, seed=997)
    # Define a random forest classifier
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=997)
    # Fit the model
    clf.fit(X_train, y_train)
    # Predict using the test datasets
    Pred = clf.predict(X_test)
    # Print accuracy
    print(round(accuracy_score(y_test, Pred), 2))
    # Save the model
    with open(os.path.dirname(os.path.abspath(__file__)) + "/../model/model.pkl", "wb") as model_file:
        joblib.dump(clf, model_file)


if __name__ == "__main__":
    main()
