from sklearn.datasets import load_iris
from sklearn import tree

classifier = tree.DecisionTreeClassifier()
iris = load_iris()
X, Y = iris.data, iris.target

classifier = classifier.fit(X,Y)

tree.plot_tree(classifier)