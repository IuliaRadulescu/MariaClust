from __future__ import division

import numpy as np
import sys
import collections
import matplotlib.pyplot as plt
import random

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.cure import cure
from pyclustering.cluster.clarans import clarans
from pyclustering.cluster.agglomerative import agglomerative, type_link

import evaluation_measures

'''
Formatul dictionarului pentru evaluare
{ clasa_0 : { cluster_0: nr_puncte, cluster_1: nr_puncte, ... cluster_n: nr_puncte}, clasa_1: { cluster_0: nr_puncte, cluster_1: nr_puncte, ... cluster_n: nr_puncte}....}

Exemplu:

Se da un dataset cu 2 clase si 400 de puncte
- clasa 1 are 150 de puncte
- clasa 2 are 250 de puncte

dupa clusterizare avem urmatoarele:
- clusterul 1 are 140 de puncte din clasa 1 si  20 de puncte din clasa 2
- clusterul 2 are  10 de puncte din clasa 1 si 230 de puncte din clasa 2

dictionarul va arata in felul urmator:
{
clasa_1 : {cluster_1: 140, cluster_2: 10},
clasa_2 : {cluster_1:  20, cluster_2: 230}
}
'''


class EvaluateAlgorithms:

	def __init__(self, no_dims):

		self.no_dims = no_dims

	def runKMeans(self, k, X):
		cluster_points = {}
		for q in range(k):
			cluster_points[q] = list()
		y_pred = KMeans(n_clusters=k, random_state=0).fit_predict(X)
		#print(y_pred)
		for point_id in range(len(X)):
			cluster_points[y_pred[point_id]].append(X[point_id])
		#print(cluster_points)
		return cluster_points
		

	def evaluate_cluster(self, clase_points, cluster_points, filename):
		
		evaluation_dict = {}
		point2cluster = {}
		point2class = {}

		idx = 0
		for elem in clase_points:
			evaluation_dict[idx] = {}
			for points in clase_points[elem]:
				point2class[points] = idx
			idx += 1

		idx = 0
		for elem in cluster_points:
			for point in cluster_points[elem]:
				index_dict = list()
				for dim in range(self.no_dims):
					index_dict.append(point[dim])
				point2cluster[tuple(index_dict)] = idx
			for c in evaluation_dict:
				evaluation_dict[c][idx] = 0
			idx += 1

		for point in point2cluster:
			evaluation_dict[point2class[point]][point2cluster[point]] += 1
				

		print('Purity: ', evaluation_measures.purity(evaluation_dict))
		print('Entropy: ', evaluation_measures.entropy(evaluation_dict)) # perfect results have entropy == 0
		print('RI ', evaluation_measures.rand_index(evaluation_dict))
		print('ARI ', evaluation_measures.adj_rand_index(evaluation_dict))

		f = open("rezultate_evaluare_KMEANS.txt", "a")
		f.write("Rezultate evaluare pentru setul de date "+str(filename)+"\n")
		f.write('Purity: '+str(evaluation_measures.purity(evaluation_dict))+"\n")
		f.write('Entropy: '+str(evaluation_measures.entropy(evaluation_dict))+"\n")
		f.write('RI: '+str(evaluation_measures.rand_index(evaluation_dict))+"\n")
		f.write('ARI: '+str(evaluation_measures.adj_rand_index(evaluation_dict))+"\n")
		f.write("\n")
		f.close()

	def random_color_scaled(self):
		b = random.randint(0, 255)
		g = random.randint(0, 255)
		r = random.randint(0, 255)
		return [round(b/255,2), round(g/255,2), round(r/255,2)]

	def plot_clusters(self, cluster_points):
		for cluster_id in cluster_points:
			color = self.random_color_scaled()
			#print(color)
			for point in cluster_points[cluster_id]:
				plt.scatter(point[0], point[1], color=color)

		plt.show()

if __name__ == "__main__":
	
	home_path = "F:\\IULIA\\GITHUB_IULIA\\MariaClust\\datasets\\"
	filenames = [home_path+"aggregation.txt", home_path+"compound.txt", home_path+"d31.txt", home_path+"flame.txt", home_path+"jain.txt", home_path+"pathbased.txt", home_path+"r15.txt", home_path+"spiral.txt"]
	no_clusters_all = [7, 6, 31, 2, 2, 3, 15, 3]
	no_dims_all = [2, 2, 2, 2, 2, 2, 2, 2]

	for nr_crt in range(len(filenames)):

		filename = filenames[nr_crt]
		no_clusters = no_clusters_all[nr_crt]
		no_dims = no_dims_all[nr_crt]

		each_dimension_values = collections.defaultdict(list)
		dataset_xy = list()
		dataset_xy_validate = list()
		clase_points = collections.defaultdict(list)

		with open(filename) as f:
				content = f.readlines()

		content = [l.strip() for l in content]

		for l in content:
			aux = l.split('\t')
			for dim in range(no_dims):
				each_dimension_values[dim].append(float(aux[dim]))
			list_of_coords = list()
			for dim in range(no_dims):
				list_of_coords.append(float(aux[dim]))
			dataset_xy.append(list_of_coords)
			dataset_xy_validate.append(int(aux[no_dims]))
			clase_points[int(aux[no_dims])].append(tuple(list_of_coords))

		evaluateAlg = EvaluateAlgorithms(no_dims)
		cluster_points = evaluateAlg.runKMeans(no_clusters, dataset_xy)
		evaluateAlg.evaluate_cluster(clase_points, cluster_points, filename)