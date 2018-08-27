import numpy as np
import sys
import csv
import matplotlib.pyplot as plt

from spherecluster import SphericalKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering



if __name__ == "__main__":
    file = sys.argv[1]
    k = int(sys.argv[2])
    l = []
    with open(file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            l.append([float(elem) for elem in row[:-1]])
    print(l)

    X = np.array(l)

    plt.figure(figsize=(12, 12))

    # K-means chior
    y_pred1 = KMeans(n_clusters=k, random_state=0).fit_predict(X)

    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred1)
    plt.title("K-Means")
    plt.show()

    # Spherical k-means
    skm = SphericalKMeans(n_clusters=k)
    y_pred2 = skm.fit(X).predict(X)

    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred2)
    plt.title("Sperical K_means")
    plt.show()
    

    # DBSCAN
    i = 1
    while i<=2:
        y_pred3 = DBSCAN(eps=i, min_samples=8).fit_predict(X)
        plt.subplot(221)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred3)
        plt.title("DBSCAN i="+str(i))
        plt.show()
        i += 0.05

    # Birch
    y_pred4 = Birch(n_clusters=k).fit(X).predict(X)
    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred4)
    plt.title("Birch")
    plt.show()

    y_pred5 = GaussianMixture(n_components=k).fit(X).predict(X)
    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred5)
    plt.title("Gaussian Mixture")
    plt.show()

    y_pred7 = SpectralClustering(n_clusters=k).fit_predict(X)
    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred7)
    plt.title("Spectral Clustering")
    plt.show()


