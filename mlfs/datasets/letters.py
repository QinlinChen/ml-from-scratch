import numpy as np
from os.path import dirname, join

def load_letters():
    """Load letters recognition dataset.
    
    Returns
    ----------
    train_set : np.ndarray
        data and labels for training

    test_set : np.ndarray
        data and labels for testing
    """

    module_path = dirname(__file__)
    data_dir = join(module_path, 'data', 'letters')
    data_file = join(data_dir, 'train_set.txt')
    test_file = join(data_dir, 'test_set.txt')

    train_set = np.loadtxt(data_file, delimiter=',')
    test_set = np.loadtxt(test_file, delimiter=',')

    return train_set, test_set
