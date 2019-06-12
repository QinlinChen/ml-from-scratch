import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlfs.datasets import load_watermelon, load_adult, load_letters
from mlfs.linear_model import LogisticRegression

def classification_report(y_true, y_pred):
    report_items = [
        ('accuracy', accuracy_score(y_true, y_pred)),
        ('micro_precision', precision_score(y_true, y_pred, average='micro')),
        ('micro_recall', recall_score(y_true, y_pred, average='micro')),
        ('micro_f1', f1_score(y_true, y_pred, average='micro')),
        ('macro_precision', precision_score(y_true, y_pred, average='macro')),
        ('macro_recall', recall_score(y_true, y_pred, average='macro')),
        ('macro_f1', f1_score(y_true, y_pred, average='macro'))
    ]
    report_str = ''
    for item in report_items:
        report_str += '{:20s}{:.4f}\n'.format(item[0], item[1])
    return report_str


if __name__ == "__main__":
    train_set, test_set = load_letters()
    X_train, y_train = train_set[:, :-1], train_set[:, -1]
    X_test, y_test = test_set[:, :-1], test_set[:, -1]

    LR = LogisticRegression()
    LR.fit(X_train, y_train)
    predicted = LR.predict(X_test)
    expected = y_test
    print(classification_report(expected, predicted))
