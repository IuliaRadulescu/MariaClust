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

from sklearn.neighbors.kde import KernelDensity

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
		minDist = 99999
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
					dist = calculate_centroid(cluster_points[(intermediary_centroids[q][0], intermediary_centroids[q][1])], cluster_points[(intermediary_centroids[p][0], intermediary_centroids[p][1])])
				#calculate_smallest_pairwise pentru jain si spiral

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
	kernel = st.gaussian_kde(values) #bw_method=
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
	kernel = st.gaussian_kde(values) #bw_method=
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
	


def calculate_average_pairwise_ponderat(cluster1, cluster2):
	
	'''
	Average link method ponderat - functia calculate_average_pairwise
		- calculeaza media ponderata a punctelor dintre doua clustere candidat
		- ponderile sunt densitatile punctelor estimate cu metoda kernel density estimation
	'''

	average_pairwise = 0
	sum_pairwise = 0
	sum_ponderi = 0

	for pixel1 in cluster1:
		for pixel2 in cluster2:
			distBetween = DistFunc(pixel1, pixel2)
			#print("abs = "+str(abs(pixel1[3]-pixel2[3])))
			sum_pairwise = sum_pairwise + abs(pixel1[3]-pixel2[3])*distBetween
			sum_ponderi = sum_ponderi + abs(pixel1[3]-pixel2[3])

	average_pairwise = sum_pairwise/sum_ponderi
	return average_pairwise


def calculate_average_pairwise_ponderat2(cluster1, cluster2):
	
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
			sum_pairwise = sum_pairwise + distBetween
			sum_ponderi = sum_ponderi + 1
			nr = nr+1
	average_pairwise = sum_pairwise/sum_ponderi

	sum_pdf_pixel1 = 0
	for pixel1 in cluster1:
		sum_pdf_pixel1 = sum_pdf_pixel1 + pixel1[3]

	sum_pdf_pixel2 = 0
	for pixel2 in cluster2:
		sum_pdf_pixel2 = sum_pdf_pixel2 + pixel2[3]

	average_pairwise = average_pairwise + (sum_pdf_pixel1 - sum_pdf_pixel2)*2

	return average_pairwise

def calculate_centroid_density(cluster1, cluster2):

	centroid1 = centroid_density(cluster1)
	centroid2 = centroid_density(cluster2)

	sum_powers_dens = math.pow(centroid1[0]-centroid2[0], 2) + math.pow(centroid1[1]-centroid2[1], 2) + math.pow(centroid1[2]-centroid2[2], 2)

	dist = math.sqrt(sum_powers_dens)

	return dist

def calculate_ward(cluster1, cluster2):

	centroid1 = centroid(cluster1)
	centroid2 = centroid(cluster2)

	dist = ( (len(cluster1)*len(cluster2)) / (len(cluster1) + len(cluster2)) )*(DistFunc(centroid1, centroid2)**2)

	return dist

def calculate_average_pairwise(cluster1, cluster2):
	
	'''
	Average link method ponderat - functia calculate_average_pairwise
		- calculeaza media ponderata a punctelor dintre doua clustere candidat
		- ponderile sunt densitatile punctelor estimate cu metoda kernel density estimation
	'''

	average_pairwise = 0
	sum_pairwise = 0
	sum_ponderi = 0

	for pixel1 in cluster1:
		for pixel2 in cluster2:
			distBetween = DistFunc(pixel1, pixel2)
			#print("abs = "+str(abs(pixel1[3]-pixel2[3])))
			sum_pairwise = sum_pairwise + distBetween
			sum_ponderi = sum_ponderi + 1

	average_pairwise = sum_pairwise/sum_ponderi
	return average_pairwise

def calculate_centroid(cluster1, cluster2):
	centroid1 = centroid(cluster1)
	centroid2 = centroid(cluster2)

	dist = DistFunc(centroid1, centroid2)

	return dist

def centroid(pixels):
	
	sum_red = 0;
	sum_green = 0;

	for pixel in pixels:
		
		sum_red = sum_red + pixel[0]
		sum_green = sum_green + pixel[1]
		

	red = round(sum_red/len(pixels),2)
	green = round(sum_green/len(pixels),2)

	return (red, green)

def centroid_density(pixels):
	
	sum_red = 0
	sum_green = 0
	sum_blue = 0

	for pixel in pixels:
		
		sum_red = sum_red + pixel[0]
		sum_green = sum_green + pixel[1]
		sum_blue = sum_blue + pixel[3]
		

	red = round(sum_red/len(pixels),2)
	green = round(sum_green/len(pixels),2)
	blue = round(sum_blue/len(pixels),2)


	return (red, green, blue)

def points_equal(x, y):
	if((x[0]==y[0]) and (x[1]==y[1])):
		return True
	else:
		return False

def outliers_z_score(ys):
	threshold = 3

	mean_y = np.mean(ys)
	stdev_y = np.std(ys)
	z_scores = [(y - mean_y) / stdev_y for y in ys]
	return np.where(np.abs(z_scores) > threshold)
	
def outliers_iqr(ys):
	quartile_1, quartile_3 = np.percentile(ys, [25, 75])
	iqr = quartile_3 - quartile_1
	lower_bound = quartile_1 - (iqr * 1.5)
	upper_bound = quartile_3 + (iqr * 1.5)
	outliers_iqr = list()
	for idx in range(len(ys)):
		if ys[idx] > upper_bound:
			outliers_iqr.append(idx)
		if ys[idx] < lower_bound:
			outliers_iqr.append(idx)
	return outliers_iqr

def get_closest_mean(dataset_k):

	k=3
	distances = list()
	for point in dataset_k:
		deja_parsati = list()
		while(k>0):
			neigh_id = 0
			minDist = 99999
			for id_point_k in range(len(dataset_k)):
				point_k = dataset_k[id_point_k]
				if(point_k not in deja_parsati):
					dist = DistFunc(point, point_k)
					if(dist < minDist and dist > 0):
						minDist = dist
						neigh_id = id_point_k
			distances.append(minDist)
			neigh = dataset_k[neigh_id]
			deja_parsati.append(neigh)
			k=k-1
	return sum(distances)/len(distances)

def get_closestk_neigh(point, dataset_k, id_point):
	#print("len dataset "+str(len(dataset_k)))
	#print("init point "+str(point)+" id point "+str(id_point))

	neigh_ids = list()
	distances = list()
	deja_parsati = list()
	pot_continua = 1
	closest_mean = get_closest_mean(dataset_k)
	while(pot_continua==1):
		minDist = 99999
		neigh_id = 0
		for id_point_k in range(len(dataset_k)):
			point_k = dataset_k[id_point_k]
			if(point_k not in deja_parsati):
				dist = DistFunc(point, point_k)
				if(dist < minDist and dist > 0):
					minDist = dist
					neigh_id = id_point_k
		if(len(distances)>1):
			if(minDist <= 2*closest_mean):
				neigh = dataset_k[neigh_id]
				neigh_ids.append([neigh_id, neigh])
				distances.append(minDist)
				
				deja_parsati.append(neigh)
				#print("intra")
			else:
				pot_continua = 0
				#helper = np.array(distances)
				#ceva = np.sqrt(np.mean(abs(helper - helper.mean())**2))
				#print("ceva "+str(ceva))
				print("nu mai pot continua"+str(minDist)+" "+str(closest_mean))
		else:
			neigh = dataset_k[neigh_id]
			neigh_ids.append([neigh_id, neigh])
			distances.append(minDist)
			
			deja_parsati.append(neigh)

	neigh_ids.sort(key=lambda x: x[1])

	neigh_ids_final = [n_id[0] for n_id in neigh_ids]

	'''for pixel_id in range(len(dataset_k)):
		pixel = dataset_k[pixel_id]
		if(pixel_id in neigh_ids_final):
			plt.scatter(pixel[0], pixel[1], color="r")
		else:
			plt.scatter(pixel[0], pixel[1], color="g")

		plt.annotate(str(pixel[2])+" -- "+str(pixel[5]), (pixel[0], pixel[1]))

	plt.scatter(point[0], point[1], color="b")
	plt.annotate(str(point[2])+" -- "+str(point[5]), (point[0], point[1]))
	plt.show()'''

	

	print("len neighid fin "+str(len(neigh_ids_final)))
	return neigh_ids_final


def expand_knn(point_id):
	'''
	Extind clusterul curent 
	Iau cei mai apropiati 5 vecini ai punctului curent
	Ii adaug in cluster
	Iau cei mai apropiati 5 vecini ai punctelor urmatoare
	Cand toate punctele sunt parcurse ma opresc si incep cluster nou
	'''
	global id_cluster, clusters, pixels_partition_clusters
	point = pixels_partition_clusters[point_id]
	neigh_ids = get_closestk_neigh(point, pixels_partition_clusters, point_id)
	print("neigh ids "+str(neigh_ids))
	clusters[id_cluster].append(point)
	pixels_partition_clusters[point_id][2] = id_cluster
	pixels_partition_clusters[point_id][4] = 1
	for neigh_id in neigh_ids:
		print("vecinul "+str(neigh_id))
		if(pixels_partition_clusters[neigh_id][4]==-1):
			expand_knn(neigh_id)
		#ce am dupa o expandare
		'''for pixel_id in range(len(pixels_partition_clusters)):
			pixel = pixels_partition_clusters[pixel_id]
			if(pixel[2]!=-1):
				plt.scatter(pixel[0], pixel[1], color="r")
			else:
				plt.scatter(pixel[0], pixel[1], color="g")

		plt.show()'''
		print("----sfarsit_expandare")
		

def calculate_average_cluster(dataset_xy):
	distances = list()
	print("dataset================= "+str(dataset_xy))
	for point_a in dataset_xy:
		for point_b in dataset_xy:
			dist = DistFunc(point_a, point_b)
			if(dist > 0):
				distances.append(dist)

	avg = sum(distances)/len(distances)

	return avg

def calculage_average_interpart(part_a, part_b):
	distances = list()

	for point_a in part_a:
		for point_b in part_b:
			dist = DistFunc(point_a, point_b)
			if(dist > 0):
				distances.append(dist)

	avg = sum(distances)/len(distances)

	return avg

def calculate_smallest_pairwise(cluster1, cluster2):

	min_pairwise = 999999
	for pixel1 in cluster1:
		for pixel2 in cluster2:
			if(pixel1!=pixel2):
				distBetween = DistFunc(pixel1, pixel2)
				if(distBetween < min_pairwise):
					min_pairwise = distBetween
	return min_pairwise

def calculate_smallest_pairwise_density(cluster1, cluster2):

	min_pairwise = 999999
	for pixel1 in cluster1:
		for pixel2 in cluster2:
			if(pixel1!=pixel2):
				distBetween = DistFunc(pixel1, pixel2)
				if(distBetween < min_pairwise):
					min_pairwise = distBetween*math.fabs(pixel1[3]-pixel2[3])
	return min_pairwise

def get_furthest_points(intermediary_results_final, nrPoints):

	furthest_points = list()
	print(intermediary_results_final)
	#incep din punctul cu pdf maxim (cea mai mare densitate)
	intermediary_results_final.sort(key=lambda x: x[3])

	furthest_points.append(intermediary_results_final[0])

	for i in range(nrPoints):
		max_d_sum = 0
		max_sum_pixel = 0;	
		for pixel in intermediary_results_final:
				
			dist_anchor_sum = 0
			for anchor in furthest_points:
				pixel_np = np.array(pixel)
				anchor_np = np.array(anchor)
				d_anchor = DistFunc(pixel_np, anchor_np)
				dist_anchor_sum = dist_anchor_sum + d_anchor
			if(dist_anchor_sum > max_d_sum):
				max_d_sum = dist_anchor_sum
				max_sum_pixel = pixel
		furthest_points.append(max_sum_pixel)

	return furthest_points

'''
=============================================
ALGORITM MARIACLUST
'''
if __name__ == "__main__":
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

	#detectie si eliminare outlieri

	outliers_iqr_pdf = outliers_iqr(pdf)
	print(outliers_iqr_pdf)

	dataset_xy = [dataset_xy[i] for i in range(len(dataset_xy)) if i not in outliers_iqr_pdf]

	partition_dict = collections.defaultdict(list)
	final_partitions = collections.defaultdict(list)


	'''
	Impart punctele din setul de date in n bin-uri in functie de densitatea probabilitatii. 
	Numarul de bin-uri este numarul de clustere - 1
	'''

	pixels_per_bin, bins = np.histogram(pdf, bins=8)

	#afisare bin-uri rezultate si creare partitii - un bin = o partitie
	for idx_bin in range( (len(bins)-1) ):
		culoare = random_color_scaled()
		for idx_point in range(len(dataset_xy)):
			if(pdf[idx_point]>=bins[idx_bin] and pdf[idx_point]<bins[idx_bin+1]):
				plt.scatter(dataset_xy[idx_point][0], dataset_xy[idx_point][1], color=culoare)
				partition_dict[idx_bin].append([dataset_xy[idx_point][0], dataset_xy[idx_point][1], idx_point, pdf[idx_point]]) #mentin si id-ul punctului in setul de date si densitatea de probabilitate in acel punct - o sa am nevoie pentru clusterizarea ierarhica - criteriu de unire - average link method ponderat

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

		clusters = collections.defaultdict(list)
		id_cluster = -1

		pixels_partition_clusters = list()
		pixels_partition_anchors = list()
		just_points = list()
		
		'''pdf_partition = compute_pdf_kde(pixels_partition, x_partition, y_partition)
		for pixel_id in range(len(pixels_partition)):
			pixels_partition_anchors.append([pixels_partition[pixel_id][0], pixels_partition[pixel_id][1], -1, pdf_partition[pixel_id]])'''

		for pixel in pixels_partition:
			pixels_partition_clusters.append([pixel[0], pixel[1], -1, pdf[pixel[2]], -1, pixel[2]]) #id cluster, pdf de idx_point, deja_parsat, id_point
			just_points.append([pixel[0], pixel[1]])

		for pixel_id in range(len(pixels_partition_clusters)):
			pixel = pixels_partition_clusters[pixel_id]

			'''for pixel_id_d in range(len(pixels_partition_clusters)):

				pixel_d = pixels_partition_clusters[pixel_id_d]
				pixel_helper_d = [pixel_d[0], pixel_d[1]]
				if(pixel_helper_d in anchor_points):
					if(pixel_d[4]!=-1):
						plt.scatter(pixel_d[0], pixel_d[1], color="r")
					else:
						plt.scatter(pixel_d[0], pixel_d[1], color="g")
			plt.show()'''
				
			if(pixels_partition_clusters[pixel_id][2]==-1):
				id_cluster = id_cluster + 1
				pixels_partition_clusters[pixel_id][4] = 1
				pixels_partition_clusters[pixel_id][2] = id_cluster
				clusters[id_cluster].append(pixel)
				neigh_ids = get_closestk_neigh(pixel, pixels_partition_clusters, pixel_id)
				print("neigh ids "+str(neigh_ids))
				for neigh_id in neigh_ids:
					print("vecinul "+str(neigh_id))
					if(pixels_partition_clusters[neigh_id][2]==-1):
						pixels_partition_clusters[neigh_id][4]=1
						pixels_partition_clusters[neigh_id][2]=id_cluster
						expand_knn(neigh_id)
					print("----sfarsit_expandare")
				
		
		colors = list()
		for i in range(len(clusters)):
			color = random_color_scaled()
			colors.append(color)

		#si pentru minus 1
		color = random_color_scaled()
		colors.append(color)

		#print(colors)
		ax = plt.gca()
		ax.cla() # clear things for fresh plot
		for pixel in pixels_partition_clusters:
			#circle = plt.Circle((pixel[0], pixel[1]), eps, color='b', fill=False)
			if(pixel[2]==-1):
				plt.scatter(pixel[0], pixel[1], color=colors[len(colors)-1])
			else:
				plt.scatter(pixel[0], pixel[1], color=colors[pixel[2]])
			#ax.add_artist(circle)

		plt.show()

		inner_partitions = collections.defaultdict(list)
		inner_partitions_filtered = collections.defaultdict(list)
		part_id_inner = 0
		for i in range(len(clusters)):
			for pixel in pixels_partition_clusters:
				if(pixel[2]==i):
					inner_partitions[part_id_inner].append(pixel)
			part_id_inner = part_id_inner+1
		#adaug si zgomotul
		for pixel in pixels_partition_clusters:
			if(pixel[2]==-1):
				inner_partitions[part_id_inner].append(pixel)
				part_id_inner = part_id_inner+1


		#filter partitions
		keys_to_delete = list()
		for k in inner_partitions:
			if(len(inner_partitions[k])<=1):
				keys_to_delete.append(k)

		for k in keys_to_delete:
			del inner_partitions[k]

		part_id_filtered = 0
		for part_id_k in inner_partitions:
			inner_partitions_filtered[part_id_filtered] = inner_partitions[part_id_k]
			part_id_filtered = part_id_filtered + 1


		for part_id_inner in inner_partitions_filtered:
			final_partitions[part_id] = inner_partitions_filtered[part_id_inner]
			part_id = part_id + 1

		
	#filter partitions
	'''keys_to_delete = list()
	for k in final_partitions:
		if(len(final_partitions[k])<=1): #2 pentru celelalte in afara de aggregations
			keys_to_delete.append(k)

	for k in keys_to_delete:
		final_partitions.pop(k)'''

	#print partititons

	for k in final_partitions:
		color = random_color_scaled()
		for pixel in final_partitions[k]:
			plt.scatter(pixel[0], pixel[1], color=color)

	plt.show()


	intermediary_centroids, cluster_points = agglomerative_clustering2(final_partitions, no_clusters) #paramateri: partitiile rezultate, numarul de clustere
	print(intermediary_centroids)
	print("==============================")
	#print(cluster_points)

	plt.contourf(xx, yy, f, cmap='Blues')
	#afisare finala
	for k in cluster_points:
		c = random_color_scaled()
		for point in cluster_points[k]:
			plt.scatter(point[0], point[1], color=c)

		'''centroidpt = centroid(cluster_points[k])
		plt.scatter(centroidpt[0], centroidpt[1], color='r')'''

	plt.show()

