import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlfs.datasets import load_watermelon, load_adult, load_letters
from mlfs.linear_model import LogisticRegression
from mlfs.tree import DecisionTreeClassifier


if __name__ == "__main__":
    dataset = load_watermelon('2.0')
    X, y = dataset[:, :-1], dataset[:, -1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)
