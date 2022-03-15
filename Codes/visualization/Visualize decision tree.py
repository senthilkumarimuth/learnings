from sklearn.datasets import load_iris
from sklearn import tree

clf = tree.DecisionTreeClassifier(random_state=0)
iris = load_iris()

clf = clf.fit(iris.data, iris.target)

#Plot tree
plt.figure(figsize = [8,8])
tree.plot_tree(clf)
plt.show();
