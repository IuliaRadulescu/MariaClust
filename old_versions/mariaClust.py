from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
# import peakutils

import sys
import os
from random import randint
from random import shuffle
import math
import collections

import sklearn
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import argrelextrema
from scipy.spatial import ConvexHull

'''
=============================================
FUNCTII AUXILIARE
'''

def agglomerative_clustering2(partitions, nr_final_clusters):
	'''
	Clusterizare ierarhica aglomerativa pornind de la niste clustere (partitii) deja create
	Fiecare partitie este reprezentata de catre centroidul ei
	Identificatorii partitiilor sunt pastrati in lista intermediary_centroids, iar clusterele cu punctele asociate sunt pastrate in dictionarul cluster_points.
	cluster_points => cheile sunt identificatorii partitiilor (centroizii lor), iar valorile sunt punctele asociate
	Criteriul de unire a doua clustere este average link method ponderat
	Average link method ponderat - functia calculate_average_pairwise
		- calculeaza media ponderata a punctelor dintre doua clustere candidat
		- ponderile sunt densitatile punctelor estimate cu metoda kernel density estimation
	Motivatie utilizare medie ponderata:
	Clusterele candidat trebuie unite tinand cont si de zona de densitate in care se afla - distanta nu este un criteriu suficient.
	Astfel favorizez unirea a doua clustere din zone cu densitate asemanatoare.'''
	

	nr_clusters_agg = len(partitions)
	intermediary_centroids = list()
	#intermediary_centroids este de o lista cu identificatorii clusterelor
	
	'''
	clusterele sunt mentiunte in cluster_points
	id-ul unui cluster este centroidul lui
	'''

	#cluster_points = collections.defaultdict(list)

	cluster_points = dict()

	print("len partitions "+str(len(partitions)))

	for k in partitions:
		centroid_partition = centroid(partitions[k])
		cluster_points[(centroid_partition[0], centroid_partition[1])] = []
		intermediary_centroids.append(centroid_partition)
	
	for k in partitions:
		centroid_partition = centroid(partitions[k])
		for pixel in partitions[k]:
			cluster_points[(centroid_partition[0], centroid_partition[1])].append(pixel)

	#print cluster_points

	while(nr_clusters_agg > nr_final_clusters):
		uneste_a_idx = 0
		uneste_b_idx = 0
		minDist = 9999
		minDistancesWeights = list()
		mdw_uneste_a_idx = list()
		mdw_uneste_b_idx = list()
		for q in range(len(intermediary_centroids)):
			for p in range(q+1, len(intermediary_centroids)):
				
				#DISTANTA SINGLE LINKAGE
				#print("------nr_proces:"+str(numar_proces)+" p = "+str(p))
				#print("------nr_proces:"+str(numar_proces)+" q = "+str(q))
				#dist = calculate_smallest_pairwise_density(cluster_points[(intermediary_centroids[q][0], intermediary_centroids[q][1])], cluster_points[(intermediary_centroids[p][0], intermediary_centroids[p][1])])
				centroid_q = centroid(cluster_points[(intermediary_centroids[q][0], intermediary_centroids[q][1])])
				centroid_p = centroid(cluster_points[(intermediary_centroids[p][0], intermediary_centroids[p][1])])
				if(centroid_q!=centroid_p):
					dist = calculate_average_pairwise(cluster_points[(intermediary_centroids[q][0], intermediary_centroids[q][1])], cluster_points[(intermediary_centroids[p][0], intermediary_centroids[p][1])])
				#DISTANTA CENTROIZI

				if(dist<minDist):
					minDist = dist
					uneste_a_idx = q
					uneste_b_idx = p
					
		helperCluster = list()
		for cluster_point in cluster_points[(intermediary_centroids[uneste_a_idx][0], intermediary_centroids[uneste_a_idx][1])]:
			
			helperCluster.append(cluster_point)
		
		for cluster_point in cluster_points[(intermediary_centroids[uneste_b_idx][0], intermediary_centroids[uneste_b_idx][1])]:
			
			helperCluster.append(cluster_point)

		
		newCluster = centroid(helperCluster)

		
		del cluster_points[(intermediary_centroids[uneste_a_idx][0], intermediary_centroids[uneste_a_idx][1])]
		del cluster_points[(intermediary_centroids[uneste_b_idx][0], intermediary_centroids[uneste_b_idx][1])]

		
		cluster_points[(newCluster[0], newCluster[1])] = []
		for pointHelper in helperCluster:
			cluster_points[(newCluster[0], newCluster[1])].append(pointHelper)

		
		value_a = intermediary_centroids[uneste_a_idx]
		value_b = intermediary_centroids[uneste_b_idx]


		for cluster_point in cluster_points[(newCluster[0], newCluster[1])]:
			if(cluster_point in intermediary_centroids):
				intermediary_centroids = list(filter(lambda a: a != cluster_point, intermediary_centroids))
		
		if(value_a in intermediary_centroids):
			intermediary_centroids = list(filter(lambda a: a != value_a, intermediary_centroids))

		if(value_b in intermediary_centroids):
			intermediary_centroids = list(filter(lambda a: a != value_b, intermediary_centroids))


		'''
		if row is a list, then don't forget that deleting an element of a list will move all following elements back one place to fill the gap
		https://stackoverflow.com/questions/3392677/python-list-assignment-index-out-of-range
		---de ce am scazut 1
		'''

		intermediary_centroids.append(newCluster)

		
		nr_clusters_agg = len(cluster_points)
		

	return intermediary_centroids, cluster_points

def compute_pdf_kde(dataset_xy, x, y):
	values = np.vstack([x, y])
	kernel = st.gaussian_kde(values) #bw_method=0.01 pentru spiral
	pdf = kernel.evaluate(values)

	scott_fact = kernel.scotts_factor()
	print("who is scott? "+str(scott_fact))
	return pdf


def evaluate_pdf_kde(dataset_xy, x, y):
	xmin = min(x)-2
	xmax = max(x)+2

	ymin = min(y)-2
	ymax = max(y)+2

	# Peform the kernel density estimate
	xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
	positions = np.vstack([xx.ravel(), yy.ravel()])
	values = np.vstack([x, y])
	kernel = st.gaussian_kde(values) #bw_method=0.01 pentru spiral
	f = np.reshape(kernel(positions).T, xx.shape)
	return (f,xmin, xmax, ymin, ymax, xx, yy)


def compute_avg_dist(points):
	dist = list()
	for p in points:
		for q in points:
			if(p[0]!=q[0] and p[1]!=q[1]):
				dist.append(DistFunc(p, q))

	return (sum(dist)/len(dist))

def random_color_scaled():
	b = randint(0, 255)
	g = randint(0, 255)
	r = randint(0, 255)
	return [round(b/255,2), round(g/255,2), round(r/255,2)]

#Distanta Euclidiana dintre doua puncte 2d
def DistFunc(x, y, printF=2):
	sum_powers = math.pow(x[0]-y[0], 2) + math.pow(x[1]-y[1], 2)
	return math.sqrt(sum_powers)


def calculate_weight(cluster):
	densities = list()

	for pixel in cluster:
		densities.append(pixel[3])
	
	densities = np.array(densities)
	print("res "+str(sum(densities)/len(densities)))

	return sum(densities)/len(densities)
	


def calculate_average_pairwise(cluster1, cluster2):
	
	'''
	Average link method ponderat - functia calculate_average_pairwise
		- calculeaza media ponderata a punctelor dintre doua clustere candidat
		- ponderile sunt densitatile punctelor estimate cu metoda kernel density estimation
	'''

	average_pairwise = 0
	sum_pairwise = 0
	nr = 0
	sum_ponderi = 0

	for pixel1 in cluster1:
		for pixel2 in cluster2:
			distBetween = DistFunc(pixel1, pixel2)
			#print("abs = "+str(abs(pixel1[3]-pixel2[3])))
			sum_pairwise = sum_pairwise + abs(pixel1[3]-pixel2[3])*distBetween
			sum_ponderi = sum_ponderi + abs(pixel1[3]-pixel2[3])
			nr = nr+1
	average_pairwise = sum_pairwise/sum_ponderi
	return average_pairwise

def centroid(pixels):
	
	sum_red = 0;
	sum_green = 0;

	for pixel in pixels:
		
		sum_red = sum_red + pixel[0]
		sum_green = sum_green + pixel[1]
		

	red = round(sum_red/len(pixels),2)
	green = round(sum_green/len(pixels),2)

	return (red, green)

def points_equal(x, y):
	if((x[0]==y[0]) and (x[1]==y[1])):
		return True
	else:
		return False

def get_eps_neigh(point, dataset, eps):
	'''
	Returneaza punctele din dataset situate la distanta cel putin eps fata de punctul point
	'''
	neighs = list()
	for q in range(len(dataset)):
		if(DistFunc(dataset[q], point)<=eps and DistFunc(dataset[q], point)!=0):
			neighs.append(q)
	return neighs

def expand(point_id, eps, minPts):
	'''
	Extind clusterul curent (similar DBSCAN)
	'''
	global id_cluster, clusters, pixels_partition_clusters
	point = pixels_partition_clusters[point_id]
	neigh_ids = get_eps_neigh(point, pixels_partition_clusters, eps)
	if(len(neigh_ids) >= minPts):
		#print("len(neighs) "+str(len(neigh_ids)))
		clusters[id_cluster].append(point)
		pixels_partition_clusters[point_id][2] = id_cluster
		for neigh_id in neigh_ids:
			if(pixels_partition_clusters[neigh_id][2]==-1):
				expand(neigh_id, eps, minPts)
			pixels_partition_clusters[neigh_id][2] = id_cluster

def calculate_eps(dataset):
	minDistances=list()
	sum_ponderi = 0
	point_p = 0
	point_q = 0
	for i in range(len(dataset)):
		minDist = 9999
		for j in range(i+1, len(dataset)):
			if(i!=j):
				dist = DistFunc(dataset[i], dataset[j])
				if(dist < minDist):
					minDist = dist
					point_p = dataset[i]
					point_q = dataset[j]
		if(minDist!=9999):
			double = minDist*abs(point_p[3]-point_q[3])
			minDistances.append(double)
			sum_ponderi = sum_ponderi +abs(point_p[3]-point_q[3])
	#print(minDistances)
	minDistances = np.array(minDistances)
	eps = sum(minDistances)/sum_ponderi
	return eps 


'''
=============================================
ALGORITM MARIACLUST
'''

filename = sys.argv[1]
no_clusters = int(sys.argv[2])

with open(filename) as f:
	content = f.readlines()

content = [l.strip() for l in content]
x = list()
y = list()
dataset_xy = list()
for l in content:
	aux = l.split('\t')
	x.append(float(aux[0]))
	y.append(float(aux[1]))
	dataset_xy.append([float(aux[0]), float(aux[1])])

pdf = compute_pdf_kde(dataset_xy, x, y) #calculez functia densitate probabilitate utilizand kde
f,xmin, xmax, ymin, ymax, xx, yy = evaluate_pdf_kde(dataset_xy, x, y) #pentru afisare zone dense albastre
plt.contourf(xx, yy, f, cmap='Blues') #pentru afisare zone dense albastre

partition_dict = collections.defaultdict(list)
final_partitions = collections.defaultdict(list)

'''
Impart punctele din setul de date in n bin-uri in functie de densitatea probabilitatii. 
Numarul de bin-uri este numarul de clustere - 1
'''

pixels_per_bin, bins = np.histogram(pdf, bins=7) #!!!! numarul de binuri e nr clustere-1 (pentru Aggregation e numarul de clustere, deci 7)

#afisare bin-uri rezultate si creare partitii - un bin = o partitie
for idx_bin in range( (len(bins)-1) ):
	culoare = random_color_scaled()
	for idx_point in range(len(dataset_xy)):
		if(pdf[idx_point]>=bins[idx_bin] and pdf[idx_point]<bins[idx_bin+1]):
			plt.scatter(dataset_xy[idx_point][0], dataset_xy[idx_point][1], color=culoare)
			partition_dict[idx_bin].append([dataset_xy[idx_point][0], dataset_xy[idx_point][1], idx_point, pdf[idx_point]]) #mentin si id-ul punctului in setul de date si densitatea de probabilitate in acel punct - o sa am nevoie pentru clusterizarea ierarhica - criteriu de unire - average link method ponderat

plt.title("Rezultat intermediar, partitii initiale")
plt.show()

noise = list()
part_id=0

'''
Pasul anterior atribuie zonele care au aceeasi densitate aceluiasi cluster, chiar daca aceste zone se afla la distanta mare una fata de cealalta.
De aceea aplic un algoritm similar DBSCAN pentru a determina cat de mult se extinde o zona de densitate, si astfel partitionez zonele care se afla la distanta mare una fata de alta.
Pentru fiecare partitie creata la pasul anterior, verific pentru fiecare punct apartinand partitiei daca la o distanta eps calculata acesta are cel putin un vecin.
	- aici trebuie stabilita o formula de calcul a numarului de vecini, in general merge cu unul singur, dar in anumite cazuri are tendinta sa creeze lantisoare.
Unesc partitiile rezultate in urma separarii utilizand clusterizarea ierarhica aglomerativa modificata (utilizeaza media ponderata pentru unirea clusterelor)
	- detalii comment functie agglomerative_clustering2
Motivatie utilizare medie ponderata:
	Clusterele candidat trebuie unite tinand cont si de zona de densitate in care se afla - distanta nu este un criteriu suficient.
	Astfel favorizez unirea a doua clustere din zone cu densitate asemanatoare.
'''

for k in partition_dict:
	pixels_partition = partition_dict[k]

	x_partition = list()
	y_partition = list()
	color = random_color_scaled()
	for point in pixels_partition:
		x_partition.append(point[0])
		y_partition.append(point[1])

	'''for point in far_points:
		plt.scatter(point[0], point[1], color='r')

	for point in pixels_partition:
		if(point not in far_points):
			plt.scatter(point[0], point[1], color=color)'''

	#verific unitatea clusterelor
	minPts = 1 #1 pentru toate in afara de aggregation
	distances = list()
	for q in range(len(pixels_partition)):
		for p in range(q+1, len(pixels_partition)):
			distances.append(DistFunc(pixels_partition[q], pixels_partition[p]))
	distances.sort()
	distances = np.array(distances)
	print(distances)
	#eps = sum(distances[0:int(len(distances)/2)])/int(len(distances)/2)
	#print("eps = "+str(eps)+" minPts = "+str(minPts))
	eps = calculate_eps(pixels_partition)
	print("eps = "+str(eps)+" minPts = "+str(minPts))

	clusters = collections.defaultdict(list)
	id_cluster = 0

	pixels_partition_clusters = list()

	for pixel in pixels_partition:
		pixels_partition_clusters.append([pixel[0], pixel[1], -1, pdf[pixel[2]]]) #adica pdf de idx_point

	for pixel_id in range(len(pixels_partition_clusters)):
		if(pixels_partition_clusters[pixel_id][2]==-1):
			pixel = pixels_partition_clusters[pixel_id]
			neigh_ids = get_eps_neigh(pixel, pixels_partition_clusters, eps)
			if(len(neigh_ids) >= minPts):
				clusters[id_cluster].append(pixel)
				pixels_partition_clusters[pixel_id][2] = id_cluster
				for neigh_id in neigh_ids:
					expand(neigh_id, eps, minPts)
				id_cluster = id_cluster + 1
		
	
	colors = list()
	for i in range(len(clusters)):
		color = random_color_scaled()
		colors.append(color)

	#print(colors)
	'''ax = plt.gca()
	ax.cla() # clear things for fresh plot
	for pixel in pixels_partition_clusters:
		circle = plt.Circle((pixel[0], pixel[1]), eps, color='b', fill=False)
		plt.scatter(pixel[0], pixel[1], color=colors[pixel[2]])
		ax.add_artist(circle)

	plt.show()'''

	for i in range(len(clusters)):
		for pixel in pixels_partition_clusters:
			if(pixel[2]==i):
				final_partitions[part_id].append(pixel)
		part_id = part_id+1
	#adaug si zgomotul
	for pixel in pixels_partition_clusters:
		if(pixel[2]==-1):
			final_partitions[part_id].append(pixel)
			part_id = part_id+1
#filter partitions
keys_to_delete = list()
for k in final_partitions:
	if(len(final_partitions[k])<=1): #2 pentru celelalte in afara de aggregations
		keys_to_delete.append(k)

for k in keys_to_delete:
	final_partitions.pop(k)

#print partititons

for k in final_partitions:
	color = random_color_scaled()
	for pixel in final_partitions[k]:
		plt.scatter(pixel[0], pixel[1], color=color)

plt.title("Rezultat intermediar, partitii scindate")
plt.show()

intermediary_centroids, cluster_points = agglomerative_clustering2(final_partitions, no_clusters) #paramateri: partitiile rezultate, numarul de clustere
print(intermediary_centroids)
print("==============================")
#print(cluster_points)

plt.contourf(xx, yy, f, cmap='Blues')
#afisare finala
plt.title("Rezultat final")
for k in cluster_points:
	c = random_color_scaled()
	for point in cluster_points[k]:
		plt.scatter(point[0], point[1], color=c)

plt.show()

