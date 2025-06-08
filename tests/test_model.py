import numpy as np
import pytest
from model import stopping_distance


def test_stopping_distance_positive():
    d = stopping_distance(np.array([50.0]), np.array([1.5]), np.array([0.8]), np.array([0.0]))
    assert d[0] > 0


def test_stopping_distance_invalid():
    with pytest.raises(ValueError):
        stopping_distance(np.array([50.0]), np.array([1.0]), np.array([-0.5]), np.array([0.0]))
