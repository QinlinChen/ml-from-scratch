import numpy as np
from os.path import dirname, join


def load_watermelon(version='4.0'):
    """Load watermelon dataset.
    
    Returns
    ----------
    train_set : np.ndarray
        data and labels for training
    """

    module_path = dirname(__file__)
    data_dir = join(module_path, 'data', 'watermelon')

    if version == '4.0':
        return np.loadtxt(join(data_dir, 'watermelon_4_0.txt'))
    elif version == '3.0.a':
        return np.loadtxt(join(data_dir, 'watermelon_3_0_a.txt'))
    else:
        raise ValueError(
            'We have no such version watermelon dataset:', version)

    return None
