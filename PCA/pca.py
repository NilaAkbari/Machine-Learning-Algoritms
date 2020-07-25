import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

class PCA():
    def calculate_covariance_matrix(self, X, Y=None):
        #ماتريس covariance را حساب مي کنيم

        m = X.shape[0]
        X = X - np.mean(X, axis=0)
        Y = X if Y == None else Y - np.mean(Y, axis=0)
        return 1 / m * np.matmul(X.T, Y)

    def transform(self, X, n_components):
        # فرض مي کنيم : n=X.shape[1]و ابعاد داده را به n_component کاهش مي دهيم

        covariance_matrix = self.calculate_covariance_matrix(X)

        # مقدار ويژه و بردار ويژه را مي گيريم
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        #Sort feature vectors and take the largest top n_component group
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors[:, :n_components]

        # تبديل مي کنيم
        return np.matmul(X, eigenvectors)

#تابع اصلي
def main():
    
    # ديتاست رو load ميکنيم
    data = datasets.load_digits()
    X = data.data
    y = data.target

    # به دو کامپوننت اصلي کاهش مي دهيم
    X_trans = PCA().transform(X, 2)

    x1 = X_trans[:, 0]
    x2 = X_trans[:, 1]

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

    class_distr = []
    # کلاس هاي مختلف را plot مي کنيم.
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        _y = y[y == l]
        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

    plt.legend(class_distr, y, loc=1)

    # label
    plt.suptitle("PCA Dimensionality Reduction")
    plt.title("Digit Dataset")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


if __name__ == "__main__":
    main()
