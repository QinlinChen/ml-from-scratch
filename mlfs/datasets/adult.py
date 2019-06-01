import numpy as np
import pandas as pd
from os.path import dirname, join, exists


class AdultDataset(object):
    def __init__(self, X_train, y_train, X_test, y_test,
                 feature_names, target_names):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.target_names = target_names


def save_adult_dataset(file, dataset):
    np.savez(file, X_train=dataset.X_train, y_train=dataset.y_train,
             X_test=dataset.X_test, y_test=dataset.y_test,
             feature_names=dataset.feature_names,
             target_names=dataset.target_names)


def load_adult_dataset(file):
    with np.load(file) as data:
        return AdultDataset(data['X_train'], data['y_train'],
                            data['X_test'], data['y_test'],
                            data['feature_names'], data['target_names'])


def load_dataframe_without_unknowns(file):
    df = pd.read_csv(file, sep=r',\s*', index_col=False,
                     engine='python', na_values='?')
    return df.dropna()


def encode_data(df_data, df_test):
    cols_with_discrete_attr = ['workclass', 'education',
                               'marital-status', 'occupation', 'relationship',
                               'race', 'sex', 'native-country', 'income']

    for col in cols_with_discrete_attr:
        class_mapping = {label: idx for idx, label
                         in enumerate(set(df_data[col]))}
        df_data[col] = df_data[col].map(class_mapping)
        df_test[col] = df_test[col].map(class_mapping)


def load_adult():
    """Load adult dataset."""
    module_path = dirname(__file__)
    data_dir = join(module_path, 'data', 'adult')
    data_file = join(data_dir, 'adult.data')
    test_file = join(data_dir, 'adult.test')
    cache_file = join(data_dir, 'adult.npz')
    if exists(cache_file):
        return load_adult_dataset(cache_file)

    df_data = load_dataframe_without_unknowns(data_file)
    df_test = load_dataframe_without_unknowns(test_file)
    feature_names = list(df_data.columns)[0:-1]
    target_names = [label for label in set(df_data['income'])]

    encode_data(df_data, df_test)
    X_train = np.array(df_data.iloc[:, 0:-1])
    y_train = np.array(df_data.iloc[:, -1])
    X_test = np.array(df_test.iloc[:, 0:-1])
    y_test = np.array(df_test.iloc[:, -1])
    dataset = AdultDataset(X_train, y_train, X_test, y_test,
                           feature_names, target_names)
    save_adult_dataset(cache_file, dataset)
    return dataset


if __name__ == "__main__":
    dataset = load_adult()
    print(dataset.X_train)
    print(dataset.y_train)
