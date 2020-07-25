#استفاده از ديتا ست sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

class LDA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            # (4, 1) * (1, 4) = (4,4) -> reshape
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        # SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        # مقدار ويژه و بردار ويژه SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # براي محاسبه راحتتر transpose مي گيريم
        eigenvectors = eigenvectors.T
        
        # sort
        eigen = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[eigen]
        eigenvectors = eigenvectors[eigen]
        # n تا اولي را ذخيره مي کنيم
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminants.T)

data = datasets.load_iris()
X = data.data
y = data.target

# داده ها را به طور خطي مي گذاريم
lda = LDA(2)
lda.fit(X, y)
X_projected = lda.transform(X)

print('Shape of X:', X.shape)
print('Shape of transformed X:', X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x1, x2,
        c=y, edgecolor='none', alpha=0.8,
        cmap=plt.cm.get_cmap('viridis', 3))

plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.colorbar()
plt.show()
