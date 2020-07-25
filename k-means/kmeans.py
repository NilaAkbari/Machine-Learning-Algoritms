import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
#fasele
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
#tabe kmeans
class KMeans():

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        #ليست نمونه شاخص براي هر دسته
        self.clusters = [[] for _ in range(self.K)]
        #مرکز هردسته
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # براي بهينه کردن و بهبود دسته ها يک حلقه فور مينويسم که در بدترين
        #حالت به تعداد ماکسيمم تکرار برود
        for _ in range(self.max_iters):
            # به نزديک ترين مرکز دسته را نسبت مي دهيم و cluster مي کنيم.
            self.clusters = self._create_clusters(self.centroids)
            #رسم
            if self.plot_steps:
                self.plot()

            # از اين دسته ها مرکز هاي جديد را محاسبه مي کنيم
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # اگر مرکز ها عوض نشدند از for در مي آييم
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # نمونه ها را بسته به ايندکس کلاسترشان دسته بندي مي کنيم
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        # نمونه ها را بسته به دسته شان ليبل مي زنيم
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # نمونه ها را به نزديک ترين مرکز cluster مي کنيم
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # فاصله هر داده با کل مرکز ها را محاسبه مي کنيم و
        #نزديک ترين را انتخاب مي کنيم
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # ميانگين هر کلاستر را براي مرکز جديد انتخاب مي کنيم
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # فاصله مرکزهاي جديد با قديم
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()
