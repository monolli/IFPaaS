'''
Testing the trainer module.
'''
import numpy as np
from sklearn.datasets import load_iris

from iris import trainer


def test_data_spliter():
    # Load the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    A = trainer.data_spliter(X, y)
    B = trainer.data_spliter(X, y)
    # Assert that the default split is reproducible
    np.testing.assert_equal(A, B)
