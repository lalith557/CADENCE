import random

import numpy as np

from cadence.common.seeds import set_global_seed


def test_set_global_seed_is_deterministic_for_python_random():
    set_global_seed(1234, deterministic_torch=False)
    a = [random.random() for _ in range(5)]
    set_global_seed(1234, deterministic_torch=False)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_global_seed_is_deterministic_for_numpy():
    set_global_seed(7, deterministic_torch=False)
    a = np.random.rand(5)
    set_global_seed(7, deterministic_torch=False)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_set_global_seed_rejects_out_of_range():
    import pytest

    with pytest.raises(ValueError):
        set_global_seed(-1)
    with pytest.raises(ValueError):
        set_global_seed(2**32)
