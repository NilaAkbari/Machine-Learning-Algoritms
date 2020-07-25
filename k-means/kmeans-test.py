import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans
#براي شلوغ نشدن کد مثال را در فايل جداگانه آوردم
#تعداد مرکزها، داده ها، ويژگي ها را نوشتم
X, y = make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=42)
print(X.shape)

#تعداد clusterها را مشخص کردم
clusters = len(np.unique(y))
print(clusters)
#ماکسيمم مراحل رو 150 تعيين کردم و تابع کلاس فايل ديگر را به کار بردم
k = KMeans(K=clusters, max_iters=150, plot_steps=True)
y_pred = k.predict(X)

k.plot()
