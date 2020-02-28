'''
Testing the predictor module.
'''
import os

import numpy as np

from iris import predictor


def get_predict_type(data):
    return type(predictor.predict_iris(data, os.path.dirname(os.path.abspath(__file__)) + "/../model/model.pkl"))


def get_predict_list_type(data):
    return predictor.predict_iris(data, os.path.dirname(os.path.abspath(__file__)) + "/../model/model.pkl")[0]


def test_predict():
    assert get_predict_type([[4.8, 3.4, 1.6, 0.2], [6.5, 3.2, 5.1, 2.0]]) is list
    assert isinstance(get_predict_list_type([[4.8, 3.4, 1.6, 0.2], [6.5, 3.2, 5.1, 2.0]]),
                      type(np.array(["Hi"], dtype=str)[0]))
