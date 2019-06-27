This is my implementations of machine learning algorithms.

Its hierarchy is similar to the one of the scikit-learn.

Here are some algorithms that have been implemented:

```
mlfs --- linear_model --- LogisticRegression
      |
      |- tree --- DecisionTreeClassifier
      |
      |- ensemble --- AdaBoostClassifier
      |            `- RandomForestClassifier
      |
      |- cluster --- KMeans
      |           |- DBSCAN
      |           `- AgglomerativeClustering
      |
      `- mixture --- GaussianMixture
```
