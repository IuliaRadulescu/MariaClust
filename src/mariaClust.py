from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import sys
import os
from random import randint
from random import shuffle
import math
import collections
import evaluation_measures


'''
=============================================
FUNCTII AUXILIARE
'''

def agglomerative_clustering2(partitions, final_no_clusters, cluster_distance):
	'''
	Clusterizare ierarhica aglomerativa pornind de la niste clustere (partitii) deja create
	Fiecare partitie este reprezentata de catre centroidul ei
	Identificatorii partitiilor sunt pastrati in lista intermediary_centroids, iar clusterele cu punctele asociate sunt pastrate in dictionarul cluster_points.
	cluster_points => cheile sunt identificatorii partitiilor (centroizii lor), iar valorile sunt punctele asociate
	Criteriul de unire a doua clustere variaza'''
	no_agg_clusters = len(partitions)
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

	while(no_agg_clusters > final_no_clusters):
		uneste_a_idx = 0
		uneste_b_idx = 0
		minDist = 99999
		minDistancesWeights = list()
		mdw_uneste_a_idx = list()
		mdw_uneste_b_idx = list()
		for q in range(len(intermediary_centroids)):
			for p in range(q+1, len(intermediary_centroids)):
				
				centroid_q = centroid(cluster_points[(intermediary_centroids[q][0], intermediary_centroids[q][1])])
				centroid_p = centroid(cluster_points[(intermediary_centroids[p][0], intermediary_centroids[p][1])])
				if(centroid_q!=centroid_p):
					# calculate_smallest_pairwise pentru jain si spiral
					if(cluster_distance==1):
						dist = calculate_centroid(cluster_points[(intermediary_centroids[q][0], intermediary_centroids[q][1])], cluster_points[(intermediary_centroids[p][0], intermediary_centroids[p][1])])					
					elif(cluster_distance==2):
						dist = calculate_average_pairwise(cluster_points[(intermediary_centroids[q][0], intermediary_centroids[q][1])], cluster_points[(intermediary_centroids[p][0], intermediary_centroids[p][1])])
					elif(cluster_distance==3):
						dist = calculate_smallest_pairwise(cluster_points[(intermediary_centroids[q][0], intermediary_centroids[q][1])], cluster_points[(intermediary_centroids[p][0], intermediary_centroids[p][1])])
					elif(cluster_distance==4):
						dist = calculate_weighted_average_pairwise(cluster_points[(intermediary_centroids[q][0], intermediary_centroids[q][1])], cluster_points[(intermediary_centroids[p][0], intermediary_centroids[p][1])])
					else:
						dist = calculate_centroid(cluster_points[(intermediary_centroids[q][0], intermediary_centroids[q][1])], cluster_points[(intermediary_centroids[p][0], intermediary_centroids[p][1])])
				
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
		no_agg_clusters = len(cluster_points)
		

	return intermediary_centroids, cluster_points

def compute_pdf_kde(dataset_xy, x, y):
	'''
	Calculeaza functia probabilitate de densitate si intoarce valorile ei pentru
	punctele din dataset_xy
	'''
	values = np.vstack([x, y])
	kernel = st.gaussian_kde(values) #bw_method=
	pdf = kernel.evaluate(values)

	scott_fact = kernel.scotts_factor()
	print("who is scott? "+str(scott_fact))
	return pdf


def evaluate_pdf_kde(dataset_xy, x, y):
	'''
	Genereaza graficul in nuante de albastru pentru functia probabilitate de densitate
	calculata pentru dataset_xy
	'''
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


def random_color_scaled():
	b = randint(0, 255)
	g = randint(0, 255)
	r = randint(0, 255)
	return [round(b/255,2), round(g/255,2), round(r/255,2)]

#Distanta Euclidiana dintre doua puncte 2d
def DistFunc(x, y, printF=2):
	sum_powers = math.pow(x[0]-y[0], 2) + math.pow(x[1]-y[1], 2)
	return math.sqrt(sum_powers)

def centroid(pixels):
	
	sum_red = 0;
	sum_green = 0;

	for pixel in pixels:
		
		sum_red = sum_red + pixel[0]
		sum_green = sum_green + pixel[1]
		

	red = round(sum_red/len(pixels),2)
	green = round(sum_green/len(pixels),2)

	return (red, green)

	
def outliers_iqr(ys):
	'''
	Determina outlierii cu metoda inter-quartilelor
	'''
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
	'''
	Media distantelor celor mai apropiati k vecini pentru fiecare punct in parte

	'''
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

def get_closestk_neigh(point, dataset_k, id_point, expand_factor):
	'''
	Cei mai apropiati v vecini fata de un punct.
	Numarul v nu e constant, pentru fiecare punct ma extind cat de mult pot, adica
	atata timp cat distanta dintre punct si urmatorul vecin este mai mica decat
	expand_factor * closest_mean (closest_mean este calculata de functia anterioara)
	'''
	
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
		

		if(minDist <= expand_factor*closest_mean):
			neigh = dataset_k[neigh_id]
			neigh_ids.append([neigh_id, neigh])
			distances.append(minDist)
			
			deja_parsati.append(neigh)
		else:
			pot_continua = 0
		
	neigh_ids.sort(key=lambda x: x[1])

	neigh_ids_final = [n_id[0] for n_id in neigh_ids]

	return neigh_ids_final


def expand_knn(point_id, expand_factor):
	'''
	Extind clusterul curent 
	Iau cei mai apropiati v vecini ai punctului curent
	Ii adaug in cluster
	Iau cei mai apropiati v vecini ai celor v vecini
	Cand toate punctele sunt parcurse (toti vecinii au fost parcursi) ma opresc si incep cluster nou
	'''
	global id_cluster, clusters, pixels_partition_clusters
	point = pixels_partition_clusters[point_id]
	neigh_ids = get_closestk_neigh(point, pixels_partition_clusters, point_id, expand_factor)
	clusters[id_cluster].append(point)
	pixels_partition_clusters[point_id][2] = id_cluster
	pixels_partition_clusters[point_id][4] = 1
	if(len(neigh_ids)>0):
		for neigh_id in neigh_ids:
			
			if(pixels_partition_clusters[neigh_id][4]==-1):
				expand_knn(neigh_id, expand_factor)
		

def calculate_weighted_average_pairwise(cluster1, cluster2):
	
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
			
			sum_pairwise = sum_pairwise + abs(pixel1[3]-pixel2[3])*distBetween
			sum_ponderi = sum_ponderi + abs(pixel1[3]-pixel2[3])

	average_pairwise = sum_pairwise/sum_ponderi
	return average_pairwise


def calculate_average_pairwise(cluster1, cluster2):

	average_pairwise = 0
	sum_pairwise = 0
	nr = 0

	for pixel1 in cluster1:
		for pixel2 in cluster2:
			distBetween = DistFunc(pixel1, pixel2)
			sum_pairwise = sum_pairwise + distBetween
			nr = nr + 1

	average_pairwise = sum_pairwise/nr
	return average_pairwise

def calculate_smallest_pairwise(cluster1, cluster2):

	min_pairwise = 999999
	for pixel1 in cluster1:
		for pixel2 in cluster2:
			if(pixel1!=pixel2):
				distBetween = DistFunc(pixel1, pixel2)
				if(distBetween < min_pairwise):
					min_pairwise = distBetween
	return min_pairwise


def calculate_centroid(cluster1, cluster2):
	centroid1 = centroid(cluster1)
	centroid2 = centroid(cluster2)

	dist = DistFunc(centroid1, centroid2)

	return dist

def split_partitions(partition_dict, expand_factor):
	global id_cluster, clusters, pixels_partition_clusters,pdf

	print(expand_factor)
	noise = list()
	part_id=0
	final_partitions = collections.defaultdict(list)
	clusters = collections.defaultdict(list)

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
		
		for pixel in pixels_partition:
			pixels_partition_clusters.append([pixel[0], pixel[1], -1, pdf[pixel[2]], -1, pixel[2], pixel[4]]) #id cluster, pdf de idx_point, deja_parsat, id_point si clasa din care face parte pctul
			just_points.append([pixel[0], pixel[1]])

		for pixel_id in range(len(pixels_partition_clusters)):
			pixel = pixels_partition_clusters[pixel_id]
				
			if(pixels_partition_clusters[pixel_id][2]==-1):
				id_cluster = id_cluster + 1
				pixels_partition_clusters[pixel_id][4] = 1
				pixels_partition_clusters[pixel_id][2] = id_cluster
				clusters[id_cluster].append(pixel)
				neigh_ids = get_closestk_neigh(pixel, pixels_partition_clusters, pixel_id, expand_factor)
				
				for neigh_id in neigh_ids:
					if(pixels_partition_clusters[neigh_id][2]==-1):
						pixels_partition_clusters[neigh_id][4]=1
						pixels_partition_clusters[neigh_id][2]=id_cluster
						expand_knn(neigh_id, expand_factor)
					
				
		
		colors = list()
		for i in range(len(clusters)):
			color = random_color_scaled()
			colors.append(color)

		#si pentru minus 1
		color = random_color_scaled()
		colors.append(color)

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
				

		#filter partitions - le elimin pe cele care contin un singur punct
		keys_to_delete = list()
		for k in inner_partitions:
			if(len(inner_partitions[k])<=1):
				keys_to_delete.append(k)
				#salvam aceste puncte si le reasignam la sfarsit celui mai apropiat cluster
				if(len(inner_partitions[k])>0):
					for pinner in inner_partitions[k]:
						noise.append(pinner)
		for k in keys_to_delete:
			del inner_partitions[k]

		part_id_filtered = 0
		for part_id_k in inner_partitions:
			inner_partitions_filtered[part_id_filtered] = inner_partitions[part_id_k]
			part_id_filtered = part_id_filtered + 1


		for part_id_inner in inner_partitions_filtered:
			final_partitions[part_id] = inner_partitions_filtered[part_id_inner]
			part_id = part_id + 1

	return (final_partitions, noise)

def create_validation_dict(clase_points, cluster_points, intermediary_centroids):
	'''
	{
	clasa_1 : {cluster_1: 140, cluster_2: 10},
	clasa_2 : {cluster_1:  20, cluster_2: 230}
	}
	'''
	evaluation_dict = {}

	for clasa_pct in clase_points: #clase pcte e un dictionar care are ca iduri numerele claselor
		print("clasa pct: "+str(clasa_pct))
		clusters_dict = {}
		for centroid in intermediary_centroids:
			#pentru fiecare clasa, parcurgem clusterele unul cate unul sa vedem cate puncte avem din clasa respectiva
			pcte_clasa_in_cluster = list()
			print("=====centroid "+str(centroid))
			for pixel in cluster_points[centroid]:
				if(pixel[6]==clasa_pct):
					pcte_clasa_in_cluster.append((pixel[0], pixel[1]))

				'''
				Linia comentata se aplica in cazul in care cluster_1 este un tuplu format doar din punctele din clasa_1 care apartin de cluster_1
				Dar cred ca am inteles gresit si de fapt cluster1 este un tuplu cu toate punctele din acel cluster, ca mai jos
				
				tuplu_pcte_clasa_in_cluster = tuple(point for point in pcte_clasa_in_cluster)
				'''
				tuplu_pcte_clasa_in_cluster = tuple( (point[0], point[1]) for point in cluster_points[centroid])
				clusters_dict[tuplu_pcte_clasa_in_cluster] = len(pcte_clasa_in_cluster)			

		#verificare
		for clusterx in clusters_dict:
			print("=====nr_pcte in clasa pct in clusterul x "+str(clusters_dict[clusterx]))

		tuplu_pcte_in_clasa = tuple(point for point in clase_points[clasa_pct])
		evaluation_dict[clasa_pct] = clusters_dict

	print(evaluation_dict)
	return evaluation_dict

def evaluate_cluster(clase_points, cluster_points):
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
			point2cluster[(point[0], point[1])] = idx
		for c in evaluation_dict:
			evaluation_dict[c][idx] = 0
		idx += 1



	for point in point2class:		
		if point2cluster.get(point, -1) == -1:
			print("punct pierdut dupa clustering:", point)

	for point in point2cluster:
		evaluation_dict[point2class[point]][point2cluster[point]] += 1
			

	print('Purity:  ', evaluation_measures.purity(evaluation_dict))
	print('Entropy: ', evaluation_measures.entropy(evaluation_dict)) # perfect results have entropy == 0
	print('RI       ', evaluation_measures.rand_index(evaluation_dict))
	print('ARI      ', evaluation_measures.adj_rand_index(evaluation_dict))


'''
=============================================
ALGORITM MARIACLUST
'''
if __name__ == "__main__":
	filename = sys.argv[1]
	no_clusters = int(sys.argv[2]) #no  clusters
	no_bins = int(sys.argv[3]) #no bins
	expand_factor = float(sys.argv[4]) # expantion factor how much a cluster can expand based on the number of neighbours -- factorul cu care inmultesc closest mean (cat de mult se poate extinde un cluster pe baza vecinilor)
	cluster_distance = int(sys.argv[5])
	'''
	how you compute the dinstance between clusters:
	1 = centroid linkage
	2 = average linkage
	3 = single linkage
	4 = average linkage ponderat
	'''

	with open(filename) as f:
		content = f.readlines()

	content = [l.strip() for l in content]
	x = list()
	y = list()
	dataset_xy = list()
	dataset_xy_validate = list()
	clase_points = collections.defaultdict(list)

	for l in content:
		aux = l.split('\t')
		x.append(float(aux[0]))
		y.append(float(aux[1]))
		dataset_xy.append([float(aux[0]), float(aux[1])])
		dataset_xy_validate.append([float(aux[0]), float(aux[1]), int(aux[2])]) #aux[2] este nr cluster
		clase_points[int(aux[2])].append((float(aux[0]), float(aux[1])))

	pdf = compute_pdf_kde(dataset_xy, x, y) #calculez functia densitate probabilitate utilizand kde
	f,xmin, xmax, ymin, ymax, xx, yy = evaluate_pdf_kde(dataset_xy, x, y) #pentru afisare zone dense albastre
	plt.contourf(xx, yy, f, cmap='Blues') #pentru afisare zone dense albastre

	#detectie si eliminare outlieri

	outliers_iqr_pdf = outliers_iqr(pdf)
	print(outliers_iqr_pdf)

	dataset_xy = [dataset_xy[i] for i in range(len(dataset_xy)) if i not in outliers_iqr_pdf]

	partition_dict = collections.defaultdict(list)
	
	'''
	Impart punctele din setul de date in n bin-uri in functie de densitatea probabilitatii. 
	Numarul de bin-uri este numarul de clustere - 1
	'''

	pixels_per_bin, bins = np.histogram(pdf, bins=no_bins)

	#afisare bin-uri rezultate si creare partitii - un bin = o partitie
	for idx_bin in range( (len(bins)-1) ):
		culoare = random_color_scaled()
		for idx_point in range(len(dataset_xy)):
			if(pdf[idx_point]>=bins[idx_bin] and pdf[idx_point]<=bins[idx_bin+1]):
				plt.scatter(dataset_xy[idx_point][0], dataset_xy[idx_point][1], color=culoare)
				partition_dict[idx_bin].append( [dataset_xy[idx_point][0], dataset_xy[idx_point][1], idx_point, pdf[idx_point], dataset_xy_validate[idx_point][2]]) #mentin si id-ul punctului in setul de date, densitatea de probabilitate in acel punct si clasa din care face parte pctul 

	plt.show()


	'''
	Pasul anterior atribuie zonele care au aceeasi densitate aceluiasi cluster, chiar daca aceste zone se afla la distanta mare una fata de cealalta.
	De aceea aplic un algoritm similar DBSCAN pentru a determina cat de mult se extinde o zona de densitate, si astfel partitionez zonele care se afla la distanta mare una fata de alta.
	Unesc partitiile rezultate in urma separarii utilizand clusterizarea ierarhica aglomerativa modificata (utilizeaza media ponderata pentru unirea clusterelor)
	'''
	
	final_partitions, noise = split_partitions(partition_dict, expand_factor) #functie care scindeaza partitiile
	print("len noise "+str(len(noise)))
	#print partititons

	for k in final_partitions:
		color = random_color_scaled()
		for pixel in final_partitions[k]:
			plt.scatter(pixel[0], pixel[1], color=color)

	plt.show()


	intermediary_centroids, cluster_points = agglomerative_clustering2(final_partitions, no_clusters, cluster_distance) #paramateri: partitiile rezultate, numarul de clustere
	

	#reasignez zgomotul clasei cu cel mai apropiat vecin

	for noise_point in noise:
		#verific care e cel mai apropiat cluster de punctul noise_point
		closest_centroid = 0
		minDist = 99999
		for centroid in intermediary_centroids:
			for pixel in cluster_points[centroid]:
				dist = DistFunc(noise_point, pixel)
				if(dist < minDist):
					minDist = dist
					closest_centroid = centroid
		cluster_points[closest_centroid].append(noise_point)

	# create_validation_dict(clase_points, cluster_points, intermediary_centroids)
	evaluate_cluster(clase_points, cluster_points)
	
	print("==============================")
	

	plt.contourf(xx, yy, f, cmap='Blues')
	#afisare finala
	for k in cluster_points:
		c = random_color_scaled()
		for point in cluster_points[k]:
			plt.scatter(point[0], point[1], color=c)

	plt.show()

