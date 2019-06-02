This is my implementations of machine learning algorithms.

Its hierarchy is similar to the scikit-learn.

Here are some algorithms that implemented:

```
mlfs --- linear_model --- LogisticRegression
      |
      |- ensemble --- AdaBoostClassifer
      |            `- RandomForestClassifer
      |
      |- cluster --- KMeans
      |           |- DBSCAN
      |           `- AgglomerativeClustering
      |
      `- mixture --- GaussianMixture
```
