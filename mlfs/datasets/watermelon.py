import numpy as np
from os.path import dirname, join


def load_watermelon(version='4.0'):
    """Load watermelon dataset."""
    module_path = dirname(__file__)

    if version == '4.0':
        return np.loadtxt(join(
            module_path, 'data', 'watermelon', 'watermelon_4_0.txt'))
    else:
        raise ValueError(
            'We have no such version watermelon dataset:', version)

    return None
