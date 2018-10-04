from __future__ import division

import numpy as np
import sys
import collections
import matplotlib.pyplot as plt
import random
import math

from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
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

	def runBirch(self, k, X):
		cluster_points = {}
		for q in range(k):
			cluster_points[q] = list()
		y_pred = Birch(n_clusters=k).fit(X).predict(X)
		for point_id in range(len(X)):
			cluster_points[y_pred[point_id]].append(X[point_id])
		#print(cluster_points)
		return cluster_points

	def runGaussianMixture(self, k, X):
		cluster_points = {}
		for q in range(k):
			cluster_points[q] = list()
		y_pred = GaussianMixture(n_components=k).fit(X).predict(X)
		for point_id in range(len(X)):
			cluster_points[y_pred[point_id]].append(X[point_id])
		#print(cluster_points)
		return cluster_points

	def runSpectralClustering(self, k, X):
		cluster_points = {}
		for q in range(k):
			cluster_points[q] = list()
		y_pred = SpectralClustering(n_clusters=k).fit_predict(X)
		for point_id in range(len(X)):
			cluster_points[y_pred[point_id]].append(X[point_id])
		#print(cluster_points)
		return cluster_points

	def runCURE(self, k, X):
		cluster_points = {}
		for q in range(k):
			cluster_points[q] = list()

		cure_instance = cure(data=X, number_cluster=k)
		cure_instance.process()
		clusters = cure_instance.get_clusters()
		
		for id_point in range(len(X)):
			for cluster_id in range(len(clusters)):
				point_ids_in_cluster = [int(point_id_in_cluster) for point_id_in_cluster in  clusters[cluster_id]]
				if(id_point in point_ids_in_cluster):
					cluster_points[cluster_id].append(X[id_point])

		return cluster_points

	def runCLARANS(self, k, X):
		cluster_points = {}
		for q in range(k):
			cluster_points[q] = list()

		clarans_instance = clarans(data=X, number_clusters=k, numlocal=5, maxneighbor=5);
		clarans_instance.process();
		clusters = clarans_instance.get_clusters();
		
		for id_point in range(len(X)):
			for cluster_id in range(len(clusters)):
				point_ids_in_cluster = [int(point_id_in_cluster) for point_id_in_cluster in  clusters[cluster_id]]
				if(id_point in point_ids_in_cluster):
					cluster_points[cluster_id].append(X[id_point])

		return cluster_points

	def runOPTICS(self, X, mean_dist):
		cluster_points = {}
		y_pred = OPTICS(min_samples=3, max_eps=mean_dist).fit_predict(X)

		nr_obtained_clusters = max(y_pred)+1
		for q in range(nr_obtained_clusters): #aici numarul de clustere e valoarea maxima din y_pred
			cluster_points[q] = list()

		for point_id in range(len(X)):
			#eliminam zgomotele
			if(y_pred[point_id]!=-1):
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

		f = open("rezultate_evaluare_OPTICS.txt", "a")
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

	def DistFunc(self, x, y):

		sum_powers = 0
		for dim in range(self.no_dims):
			sum_powers = math.pow(x[dim]-y[dim], 2) + sum_powers
		return math.sqrt(sum_powers)

	def get_mean_dist(self, X):
		distances = list()
		for id_x in range(len(X)-1):
			for id_y in range(id_x+1, len(X)):
				dist = self.DistFunc(X[id_x], X[id_y])
				distances.append(dist)
		return sum(distances)/len(distances)


	def plot_clusters(self, cluster_points, algoritm, set_de_date):
		fig, ax = plt.subplots(nrows=1, ncols=1)
		for cluster_id in cluster_points:
			color = self.random_color_scaled()
			#print(color)
			for point in cluster_points[cluster_id]:
				ax.scatter(point[0], point[1], color=color)

		fig.savefig('F:\\IULIA\\GITHUB_IULIA\\MariaClust\\results\\'+str(algoritm)+"_"+str(set_de_date)+'.png')   # save the figure to file
		plt.close(fig)

if __name__ == "__main__":
	
	home_path = "F:\\IULIA\\GITHUB_IULIA\\MariaClust\\datasets\\"
	filenames = [home_path+"aggregation.txt", home_path+"compound.txt", home_path+"d31.txt", home_path+"flame.txt", home_path+"jain.txt", home_path+"pathbased.txt", home_path+"r15.txt", home_path+"spiral.txt"]
	dataset_names = ["Aggregation", "Compound", "D31", "Flame", "Jain", "Pathbased", "R15", "Spiral"]
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
		evaluateAlg.plot_clusters(cluster_points, "KMEANS", dataset_names[nr_crt])
		
		cluster_points = evaluateAlg.runBirch(no_clusters, dataset_xy)
		evaluateAlg.plot_clusters(cluster_points, "BIRCH", dataset_names[nr_crt])

		cluster_points = evaluateAlg.runGaussianMixture(no_clusters, dataset_xy)
		evaluateAlg.plot_clusters(cluster_points, "GAUSSIANMIXTURE", dataset_names[nr_crt])

		cluster_points = evaluateAlg.runSpectralClustering(no_clusters, dataset_xy)
		evaluateAlg.plot_clusters(cluster_points, "SPECTRALCLUSTERING", dataset_names[nr_crt])

		cluster_points = evaluateAlg.runCURE(no_clusters, dataset_xy)
		evaluateAlg.plot_clusters(cluster_points, "CURE", dataset_names[nr_crt])
		
		cluster_points = evaluateAlg.runCLARANS(no_clusters, dataset_xy)
		evaluateAlg.plot_clusters(cluster_points, "CLARANS", dataset_names[nr_crt])

		mean_dist = evaluateAlg.get_mean_dist(dataset_xy)
		cluster_points = evaluateAlg.runOPTICS(dataset_xy, mean_dist)
		evaluateAlg.plot_clusters(cluster_points, "OPTICS", dataset_names[nr_crt])
		#evaluateAlg.evaluate_cluster(clase_points, cluster_points, filename)