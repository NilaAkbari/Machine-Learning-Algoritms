#baraye inke code kheili shooloogh nashavad, dar yek file e digar
#mesal ro avordam


import numpy as np
#sklearn baraye dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split

#az file e ghabli ke neveshte shode bood class ra import mikonim
from random_forest import RandomForest

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = RandomForest(n_trees=3, max_depth=10)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

#deghat ra print mikonad
print ("Accuracy:", acc)