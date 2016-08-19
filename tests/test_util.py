import numpy as np
from evaluation.utils.util import construct_weights
from numpy.testing import assert_array_equal


def test_construct_weights():

    WEIGHTS = np.concatenate((np.zeros(10), np.ones(5), np.zeros(5)))
    x = np.arange(20)
    range_ = np.array([10, 14])
    weights = construct_weights(x, range_)
    assert_array_equal(weights, WEIGHTS)
