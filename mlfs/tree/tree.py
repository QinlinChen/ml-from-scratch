import numpy as np
from collections import Counter


def _calc_info_entropy(labels):
    '''Calculate the infomation entropy.'''
    entropy = 0.0
    for ctr in Counter(labels).values():
        p = ctr / len(labels)
        entropy -= p * np.log2(p)
    return entropy


def _calc_info_gain(X, y, attr):
    '''Calculate the infomation gain.'''
    info_gain = _calc_info_entropy(y)

    X_attr = X[:, attr]
    for attr_value in set(X_attr):
        mask = (X_attr == attr_value)
        info_gain -= np.mean(mask) * _calc_info_entropy(y[mask])

    return info_gain


class TreeNode(object):
    def __init__(self, best_attr):
        self.best_attr = best_attr
        self.children = {}


def _is_same(arr):
    return np.all(arr == arr[0])


def _ID3(X, y, attributes):
    '''Build an ID3 decision tree.'''
    ctr = Counter(y)

    if len(ctr) == 1:
        return TreeNode(y[0])

    if not attributes or _is_same(X[:, attributes]):
        return TreeNode(ctr.most_common(1)[0][0])

    info_gains = [_calc_info_gain(X, y, attr) for attr in attributes]
    best_attr = attributes[np.argmax(info_gains)]

    tree_node = TreeNode(best_attr)
    X_attr = X[:, best_attr]
    for attr_value in set(X_attr):
        mask = (X_attr == attr_value)
        sub_X, sub_y = X[mask], y[mask]
        if len(sub_y) > 0:
            attributes.remove(best_attr)
            tree_node.children[attr_value] = _ID3(sub_X, sub_y, attributes)
            attributes.append(best_attr)
        else:
            leaf_node = TreeNode(ctr.most_common(1)[0][0])
            tree_node.children[attr_value] = leaf_node

    return tree_node


def _ID3_predict(tree_node, feature):
    '''Predict by the ID3 decision tree.'''
    if not tree_node.children:
        return tree_node.best_attr
    next_node = tree_node.children[feature[tree_node.best_attr]]
    return _ID3_predict(next_node, feature)


def _ID3_print_tree(tree_node, depth):
    for _ in range(depth):
        print('  ', end='')
    print(tree_node.best_attr)
    for child in tree_node.children.values():
        _ID3_print_tree(child, depth + 1)


class DecisionTreeClassifier(object):
    '''A decision tree classifier which only supports data with discrete features.'''

    def __init__(self):
        self.tree_ = None

    def fit(self, X, y):
        self.tree_ = _ID3(X, y, list(range(X.shape[1])))

    def predict(self, X):
        pred = np.zeros(len(X))
        for i in range(len(X)):
            pred[i] = _ID3_predict(self.tree_, X[i])
        return pred
