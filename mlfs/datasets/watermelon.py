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
    valid_version = set(['2.0', '4.0', '3.0.a'])

    if version not in valid_version:
        raise ValueError(
            'We have no such version of watermelon dataset:', version)

    data_path = join(data_dir, 'watermelon_%s.txt' % version.replace('.', '_'))
    return np.loadtxt(data_path)
