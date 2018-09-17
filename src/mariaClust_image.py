from __future__ import division

import cv2

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
	global no_dims

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
		idx_dict = list()
		for dim in range(no_dims):
			idx_dict.append(centroid_partition[dim])
		cluster_points[tuple(idx_dict)] = []
		intermediary_centroids.append(centroid_partition)
	
	for k in partitions:
		centroid_partition = centroid(partitions[k])
		idx_dict = list()
		for dim in range(no_dims):
			idx_dict.append(centroid_partition[dim])
		for pixel in partitions[k]:
			cluster_points[tuple(idx_dict)].append(pixel)

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
				idx_dict_q = list()
				idx_dict_p = list()
				for dim in range(no_dims):
					idx_dict_q.append(intermediary_centroids[q][dim])
					idx_dict_p.append(intermediary_centroids[p][dim])

				centroid_q = centroid(cluster_points[tuple(idx_dict_q)])
				centroid_p = centroid(cluster_points[tuple(idx_dict_p)])
				if(centroid_q!=centroid_p):
					# calculate_smallest_pairwise pentru jain si spiral
					if(cluster_distance==1):
						dist = calculate_centroid(cluster_points[tuple(idx_dict_q)], cluster_points[tuple(idx_dict_p)])					
					elif(cluster_distance==2):
						dist = calculate_average_pairwise(cluster_points[tuple(idx_dict_q)], cluster_points[tuple(idx_dict_p)])
					elif(cluster_distance==3):
						dist = calculate_smallest_pairwise(cluster_points[tuple(idx_dict_q)], cluster_points[tuple(idx_dict_p)])
					elif(cluster_distance==4):
						dist = calculate_weighted_average_pairwise(cluster_points[tuple(idx_dict_q)], cluster_points[tuple(idx_dict_p)])
					else:
						dist = calculate_centroid(cluster_points[tuple(idx_dict_q)], cluster_points[tuple(idx_dict_p)])
				
				if(dist<minDist):
					minDist = dist
					uneste_a_idx = q
					uneste_b_idx = p
					
		helperCluster = list()

		idx_uneste_a = list()
		idx_uneste_b = list()

		for dim in range(no_dims):
			idx_uneste_a.append(intermediary_centroids[uneste_a_idx][dim])
			idx_uneste_b.append(intermediary_centroids[uneste_b_idx][dim])

		for cluster_point in cluster_points[tuple(idx_uneste_a)]:
			helperCluster.append(cluster_point)
		
		for cluster_point in cluster_points[tuple(idx_uneste_b)]:
			helperCluster.append(cluster_point)

		
		newCluster = centroid(helperCluster)

		
		del cluster_points[tuple(idx_uneste_a)]
		del cluster_points[tuple(idx_uneste_b)]

		idx_cluster = list()
		for dim in range(no_dims):
			idx_cluster.append(newCluster[dim])

		cluster_points[tuple(idx_cluster)] = []
		for pointHelper in helperCluster:
			cluster_points[tuple(idx_cluster)].append(pointHelper)

		
		value_a = intermediary_centroids[uneste_a_idx]
		value_b = intermediary_centroids[uneste_b_idx]


		for cluster_point in cluster_points[tuple(idx_cluster)]:
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

def compute_pdf_kde(dataset_xy, each_dimension_values):
	'''
	Calculeaza functia probabilitate de densitate si intoarce valorile ei pentru
	punctele din dataset_xy
	'''
	stacking_list = list()
	for dim_id in each_dimension_values:
		stacking_list.append(each_dimension_values[dim_id])
	values = np.vstack(stacking_list)
	kernel = st.gaussian_kde(values) #bw_method=
	pdf = kernel.evaluate(values)

	scott_fact = kernel.scotts_factor()
	print("who is scott? "+str(scott_fact))
	return pdf


def evaluate_pdf_kde(dataset_xy, each_dimension_values):
	'''
	Functioneaza doar pentru doua dimensiuni
	Genereaza graficul in nuante de albastru pentru functia probabilitate de densitate
	calculata pentru dataset_xy
	'''
	x = list()
	y = list()

	x = each_dimension_values[0]
	y = each_dimension_values[1]

	xmin = min(x)-2
	xmax = max(x)+2

	ymin = min(y)-2
	ymax = max(y)+2

	# Peform the kernel density estimate
	xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
	positions = np.vstack([xx.ravel(), yy.ravel()])
	values = np.vstack([x, y])
	kernel = st.gaussian_kde(values) #bw_method=

	scott_fact = kernel.scotts_factor()
	print("who is scott eval? "+str(scott_fact))

	f = np.reshape(kernel(positions).T, xx.shape)
	return (f,xmin, xmax, ymin, ymax, xx, yy)


def random_color_scaled():
	b = randint(0, 255)
	g = randint(0, 255)
	r = randint(0, 255)
	return [round(b/255,2), round(g/255,2), round(r/255,2)]

#Distanta Euclidiana dintre doua puncte 2d
def DistFunc(x, y):
	global no_dims
	sum_powers = 0
	for dim in range(no_dims):
		sum_powers = math.pow(x[dim]-y[dim], 2) + sum_powers
	return math.sqrt(sum_powers)

def centroid(pixels):
	global no_dims
	sum_each_dim = {}
	for dim in range(no_dims):
		sum_each_dim[dim] = 0

	for pixel in pixels:
		for dim in range(no_dims):
			sum_each_dim[dim] = sum_each_dim[dim] + pixel[dim]
	
	centroid_coords = list()
	for sum_id in sum_each_dim:
		centroid_coords.append(round(sum_each_dim[sum_id]/len(pixels), 2))

	centroid_coords = tuple(centroid_coords)

	return centroid_coords

	
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
	global no_dims

	just_pdfs = [point[no_dims+1] for point in dataset_k]
	just_pdfs = list(set(just_pdfs))

	mean_pdf = sum(just_pdfs)/len(just_pdfs)

	k=int(math.ceil(0.1*len(dataset_k)))
	distances = list()
	for point in dataset_k:
		deja_parsati = list()
		if(point[no_dims+1] > mean_pdf):
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
	distances = list(set(distances))
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
	global id_cluster, clusters, pixels_partition_clusters, no_dims
	point = pixels_partition_clusters[point_id]
	neigh_ids = get_closestk_neigh(point, pixels_partition_clusters, point_id, expand_factor)
	clusters[id_cluster].append(point)
	if(len(neigh_ids)>0):
		pixels_partition_clusters[point_id][no_dims] = id_cluster
		pixels_partition_clusters[point_id][no_dims+2] = 1
		for neigh_id in neigh_ids:
			
			if(pixels_partition_clusters[neigh_id][no_dims+2]==-1):
				expand_knn(neigh_id, expand_factor)
	else:
		pixels_partition_clusters[point_id][no_dims] = -1
		pixels_partition_clusters[point_id][no_dims+2] = 1
		

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
			
			sum_pairwise = sum_pairwise + abs(pixel1[no_dims+1]-pixel2[no_dims+1])*distBetween
			sum_ponderi = sum_ponderi + abs(pixel1[no_dims+1]-pixel2[no_dims+1])

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
	global id_cluster, clusters, pixels_partition_clusters, pdf, no_dims

	print(expand_factor)
	noise = list()
	part_id=0
	final_partitions = collections.defaultdict(list)
	clusters = collections.defaultdict(list)

	for k in partition_dict:
		pixels_partition = partition_dict[k]

		clusters = collections.defaultdict(list)
		id_cluster = -1

		pixels_partition_clusters = list()
		pixels_partition_anchors = list()

		pixels_partition_clusters = pixels_partition

		for pixel_id in range(len(pixels_partition_clusters)):
			pixel = pixels_partition_clusters[pixel_id]
			
			if(pixels_partition_clusters[pixel_id][no_dims]==-1):
				id_cluster = id_cluster + 1
				pixels_partition_clusters[pixel_id][no_dims+2] = 1
				pixels_partition_clusters[pixel_id][no_dims] = id_cluster
				clusters[id_cluster].append(pixel)
				neigh_ids = get_closestk_neigh(pixel, pixels_partition_clusters, pixel_id, expand_factor)
				
				for neigh_id in neigh_ids:
					if(pixels_partition_clusters[neigh_id][no_dims]==-1):
						pixels_partition_clusters[neigh_id][no_dims+2]=1
						pixels_partition_clusters[neigh_id][no_dims]=id_cluster
						expand_knn(neigh_id, expand_factor)
					
		inner_partitions = collections.defaultdict(list)
		inner_partitions_filtered = collections.defaultdict(list)
		part_id_inner = 0
		for i in range(len(clusters)):
			for pixel in pixels_partition_clusters:
				if(pixel[no_dims]==i):
					inner_partitions[part_id_inner].append(pixel)
			part_id_inner = part_id_inner+1
		#adaug si zgomotul
		for pixel in pixels_partition_clusters:
			if(pixel[no_dims]==-1):
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
	global no_dims

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
			for dim in range(no_dims):
				index_dict.append(point[dim])
			point2cluster[tuple(index_dict)] = idx
		for c in evaluation_dict:
			evaluation_dict[c][idx] = 0
		idx += 1

	'''for point in point2class:		
		if point2cluster.get(point, -1) == -1:
			print("punct pierdut dupa clustering:", point)'''

	for point in point2cluster:
		evaluation_dict[point2class[point]][point2cluster[point]] += 1
			

	print('Purity:  ', evaluation_measures.purity(evaluation_dict))
	print('Entropy: ', evaluation_measures.entropy(evaluation_dict)) # perfect results have entropy == 0
	print('RI	   ', evaluation_measures.rand_index(evaluation_dict))
	print('ARI	  ', evaluation_measures.adj_rand_index(evaluation_dict))


'''
=============================================
ALGORITM MARIACLUST
'''
if __name__ == "__main__":
	filename = sys.argv[1]
	no_clusters = int(sys.argv[2]) #no clusters
	no_bins = int(sys.argv[3]) #no bins
	expand_factor = float(sys.argv[4]) # expantion factor how much a cluster can expand based on the number of neighbours -- factorul cu care inmultesc closest mean (cat de mult se poate extinde un cluster pe baza vecinilor)
	cluster_distance = int(sys.argv[5])
	no_dims = int(sys.argv[6]) #no dims
	'''
	how you compute the dinstance between clusters:
	1 = centroid linkage
	2 = average linkage
	3 = single linkage
	4 = average linkage ponderat
	'''

	#citire fisier imagine cu opencv

	each_dimension_values = collections.defaultdict(list)
	dataset_xy = list()

	img = cv2.imread(filename,cv2.IMREAD_COLOR)
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	image_width = len(img_rgb[0])
	image_height = len(img_rgb)
	dataset_xy = np.reshape(img_rgb, (image_width*image_height, 3))

	no_dims = 3

	for pixel in dataset_xy:
		for dim in range(no_dims):
			each_dimension_values[dim].append(float(pixel[dim]))
		
	pdf = compute_pdf_kde(dataset_xy, each_dimension_values) #calculez functia densitate probabilitate utilizand kde
	#detectie si eliminare outlieri

	outliers_iqr_pdf = outliers_iqr(pdf)
	print("Am identificat urmatorii outlieri: ")
	for outlier_id in outliers_iqr_pdf:
		print(dataset_xy[outlier_id])
	print("======================================")

	dataset_xy_aux = list()
	each_dimension_values_aux = collections.defaultdict(list)

	#eliminare outlieri, refac dataset_xy, x si y

	for q in range(len(dataset_xy)):
		if(q not in outliers_iqr_pdf):
			dataset_xy_aux.append(dataset_xy[q])
			for dim in range(no_dims):
				each_dimension_values_aux[dim].append(dataset_xy[q][dim])

	dataset_xy = dataset_xy_aux
	each_dimension_values = each_dimension_values_aux

	#recalculez pdf, ca altfel se produc erori

	pdf = compute_pdf_kde(dataset_xy, each_dimension_values) #calculez functia densitate probabilitate din nou
	if(no_dims==2):
		#coturul cu albastru este plotat doar pentru 2 dimensiuni
		f,xmin, xmax, ymin, ymax, xx, yy = evaluate_pdf_kde(dataset_xy, each_dimension_values) #pentru afisare zone dense albastre
		plt.contourf(xx, yy, f, cmap='Blues') #pentru afisare zone dense albastre
		
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
				element_to_append = list()
				for dim in range(no_dims):
					element_to_append.append(dataset_xy[idx_point][dim])
				element_to_append.append(-1) #clusterul nearest neighbour din care face parte punctul
				element_to_append.append(pdf[idx_point])
				element_to_append.append(-1) #daca punctul e deja parsta nearest neighbour
				element_to_append.append(idx_point)
				partition_dict[idx_bin].append(element_to_append)
				#scatter doar pentru 2 sau 3 dimensiuni
				if(no_dims == 2):
					plt.scatter(dataset_xy[idx_point][0], dataset_xy[idx_point][1], color=culoare)
				elif(no_dims == 3):
					plt.scatter(dataset_xy[idx_point][0], dataset_xy[idx_point][1], dataset_xy[idx_point][2], color=culoare)
	if(no_dims == 2 or no_dims == 3):
		plt.show()


	'''
	Pasul anterior atribuie zonele care au aceeasi densitate aceluiasi cluster, chiar daca aceste zone se afla la distanta mare una fata de cealalta.
	De aceea aplic un algoritm similar DBSCAN pentru a determina cat de mult se extinde o zona de densitate, si astfel partitionez zonele care se afla la distanta mare una fata de alta.
	Unesc partitiile rezultate in urma separarii utilizand clusterizarea ierarhica aglomerativa modificata (utilizeaza media ponderata pentru unirea clusterelor)
	'''
	
	final_partitions, noise = split_partitions(partition_dict, expand_factor) #functie care scindeaza partitiile
	
	if(no_dims==2):
		for k in final_partitions:
			color = random_color_scaled()
			for pixel in final_partitions[k]:
				plt.scatter(pixel[0], pixel[1], color=color)

		plt.show()


	intermediary_centroids, cluster_points = agglomerative_clustering2(final_partitions, no_clusters, cluster_distance) #paramateri: partitiile rezultate, numarul de clustere
	
	print(intermediary_centroids)
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


	for centroid in cluster_points:
		for pixel in cluster_points[centroid]:
			if(pixel in dataset_xy):
				idx_to_change = dataset_xy.index(pixel)
				dataset_xy[idx_to_change] = centroid

	img_final = np.reshape(dataset_xy, (image_height, image_width, 3))
	img_final = cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR)

	cv2.imshow('image',img_final)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

