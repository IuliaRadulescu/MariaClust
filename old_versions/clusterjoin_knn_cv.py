'''
Clusterjoin CU 2D hashing
'''

from __future__ import division


import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

import numpy as np 

import sys
import os
from random import randint
from random import shuffle
import math
import collections
import ctypes as c

from multiprocessing import Process, Pool
import multiprocessing as mp

import time

__author__ = "Iulia Radulescu"
__copyright__ = "Copyright 2015, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "radulescuiuliamaria@yahoo.com"
__status__ = "Production"

def random_color():
	b = randint(0, 255)
	g = randint(0, 255)
	r = randint(0, 255)
	return [b, g, r]

def random_color_scaled():
	b = randint(0, 255)
	g = randint(0, 255)
	r = randint(0, 255)
	return [round(b/255,2), round(g/255,2), round(r/255,2)]

def unique_rows(a):
	a = np.ascontiguousarray(a)
	unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
	return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def agglomerative_clustering2(intermediary_centroids, nr_final_clusters):
	nr_clusters = len(intermediary_centroids)
	#intermediary_centroids este de fapt nr intermediar de clustere
	#print("Pornim de la "+str(nr_clusters))

	cluster_points = collections.defaultdict(list)

	for pixel in intermediary_centroids:
		cluster_points[(pixel[0], pixel[1], pixel[2])].append(pixel)

	while(nr_clusters > nr_final_clusters):
		uneste_a_idx = 0
		uneste_b_idx = 0
		minDist = 9999
		for q in range(len(intermediary_centroids)):
			for p in range(q+1, len(intermediary_centroids)):
				centroid_q = centroid(cluster_points[(intermediary_centroids[q][0], intermediary_centroids[q][1], intermediary_centroids[q][2])])
				centroid_p = centroid(cluster_points[(intermediary_centroids[p][0], intermediary_centroids[p][1], intermediary_centroids[p][2])])
				dist = DistFunc(centroid_p, centroid_q)
				if(dist<minDist):
					minDist = dist
					uneste_a_idx = p
					uneste_b_idx = q

		helperCluster = list()
		for cluster_point in cluster_points[(intermediary_centroids[uneste_a_idx][0], intermediary_centroids[uneste_a_idx][1], intermediary_centroids[uneste_a_idx][2])]:
			helperCluster.append(cluster_point)
		
		for cluster_point in cluster_points[(intermediary_centroids[uneste_b_idx][0], intermediary_centroids[uneste_b_idx][1], intermediary_centroids[uneste_b_idx][2])]:
			helperCluster.append(cluster_point)

		newCluster = centroid(helperCluster)

		#cluster_points[(newCluster[0], newCluster[1], newCluster[2])].append(intermediary_centroids[uneste_a_idx])
		#cluster_points[(newCluster[0], newCluster[1], newCluster[2])].append(intermediary_centroids[uneste_b_idx])
		for pointHelper in helperCluster:
			cluster_points[(newCluster[0], newCluster[1], newCluster[2])].append(pointHelper)
		

		intermediary_centroids.pop(uneste_a_idx)
		intermediary_centroids.pop(uneste_b_idx)
		intermediary_centroids.append(newCluster)
		newCluster_idx = len(intermediary_centroids)-1	

		nr_clusters = nr_clusters-1
		#print(nr_clusters)

	return intermediary_centroids




#the Euclidean distance between two RGB pixels 
'''def DistFunc(a, b):
	#print (np.power((a-b), 2));
	#return round(np.sqrt( np.sum( np.power(( np.subtract(a, b)), 2), 0) ), 2)
	return np.linalg.norm(a-b)
'''


#the Euclidean distance between two RGB pixels 
def DistFunc(x, y, printF=2):
	#print (np.power((a-b), 2));
	#return round(np.sqrt( np.sum( np.power(( np.subtract(a, b)), 2), 0) ), 2)
	#return np.linalg.norm(x-y)
	#return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)])
	if(printF==1):
		#print("-------------------"+str(printF))
		print("~~~~~~~~~~~~~~")
		print(str(x[0])+" "+str(y[0]))
		print(str(x[1])+" "+str(y[1]))
		print(str(x[2])+" "+str(y[2]))
	sum_powers = math.pow(int(x[0])-int(y[0]), 2) + math.pow(int(x[1])-int(y[1]), 2) + math.pow(int(x[2])-int(y[2]), 2)
	if(printF==1):
		print("RES "+str(math.sqrt(sum_powers)))

	return math.sqrt(sum_powers)

def HashPartition(anchor_partition, partition_dict):
	global total_anchors
	#anchor_partition contine toti pixelii partitiei respective
	#print("total_anchors "+str(total_anchors));
	
	matrixDim = len(anchor_partition)
	pixelToRow = collections.defaultdict(list)
	pixelToColumn = collections.defaultdict(list)
	shuffle(anchor_partition) #randomizez ordinea pixelilor 
	#asignez random pixelii randurilor
	for row in range(matrixDim):
		pixelToRow[row] = anchor_partition[row] #de la 0 la matrixDim. oricum am randomizat ordinea lor
	shuffle(anchor_partition) #randomizez ordinea pixelilor 
	#asignez random pixelii coloanelor
	for column in range(matrixDim):
		pixelToColumn[column] = anchor_partition[column] #de la 0 la matrixDim. oricum am randomizat ordinea lor
	#partitionam matricea in 4 bucati (regiuni) aproximativ egale
	#fiecare regiune are un id de partite
	region1 = total_anchors
	region2 = total_anchors + 1
	region3 = total_anchors + 2
	region4 = total_anchors + 3
	total_anchors = total_anchors + 4

	#print("reg1"+str(region1))
	#print("reg2"+str(region2))
	#print("reg3"+str(region3))
	#print("reg4"+str(region4))

	half = int(math.floor(matrixDim/2))

	region1_row = range(0, half)
	region1_column = range(0, half)
	region2_row = range(0, half)
	region2_column = range(half, matrixDim)
	region3_row = range(half, matrixDim)
	region3_column = range(0, half)
	region4_row = range(half, matrixDim)
	region4_column = range(half, matrixDim)

	#asignam pixelii de pe randuri regiunilor (partitiilor)
	#pixelii de pe randurile care intersecteaza o regiune sunt asignati acelei regiuni
	
	for row in range(matrixDim):
		if(row in region1_row):
			partition_dict[region1].append(pixelToRow[row])
		if(row in region2_row):
			partition_dict[region2].append(pixelToRow[row])
		if(row in region3_row):
			partition_dict[region3].append(pixelToRow[row])
		if(row in region4_row):
			partition_dict[region4].append(pixelToRow[row])

	#acelasi algoritm si pentru coloane

	#asignam pixelii de pe coloane partitiilor
	
	'''for column in range(matrixDim):
		if(column in region1_column):
			partition_dict[region1].append(pixelToColumn[column])
		if(column in region2_column):
			partition_dict[region2].append(pixelToColumn[column])
		if(column in region3_column):
			partition_dict[region3].append(pixelToColumn[column])
		if(column in region4_column):
			partition_dict[region4].append(pixelToColumn[column])'''

	print("regiunea 1 contine "+str(len(partition_dict[region1])))
	print("regiunea 2 contine "+str(len(partition_dict[region2])))
	print("regiunea 3 contine "+str(len(partition_dict[region3])))
	print("regiunea 4 contine "+str(len(partition_dict[region4])))
	

def centroid(pixels):
	
	sum_red = 0;
	sum_green = 0;
	sum_blue = 0;

	for pixel in pixels:
		
		sum_red = sum_red + pixel[0]
		sum_green = sum_green + pixel[1]
		sum_blue = sum_blue + pixel[2]

	red = int( (sum_red/len(pixels)))
	green = int( (sum_green/len(pixels)))
	blue = int( (sum_blue/len(pixels)))

	return (red, green, blue)




def get_furthest_points(intermediary_results_final, nrPoints):

	furthest_points = list()

	#firts_anchor_idx = randint(0, len(intermediary_results_final)-1)
	#firts_anchor_idx = int(len(intermediary_results_final)/2)
	#furthest_points.append(intermediary_results_final[firts_anchor_idx])

	furthest_points.append(centroid(intermediary_results_final))

	for i in range(nrPoints):
		max_d_sum = 0
		max_sum_pixel = [0, 0, 0];	
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


def map_function(arg_list):
	image_to_cluster = arg_list[0]
	anchor_dict = arg_list[1]
	partition_dict = arg_list[2]
	eps = arg_list[3]

	'''print("anchor_dict = ")
	print(anchor_dict)
	print("-----------------------")
	print("partition_dict = ")
	print(partition_dict)
	print("-----------------------")'''

	for pixel in image_to_cluster:
		#care este cea mai apropiata partitie de pixel
		minDist = 99999
		closestAnchorIndex = -1
		for anchor_index in anchor_dict:
			pixel_np = np.array(pixel)
			anchor_np = np.array(anchor_dict[anchor_index])
			distAnchorPixel = DistFunc(pixel_np, anchor_np)
			if(distAnchorPixel<minDist):
				minDist = distAnchorPixel
				closestAnchorIndex = anchor_index
		partition_dict[closestAnchorIndex].append(pixel)

		'''for anchor_index in anchor_dict:
			delta = DistFunc(anchor_dict[closestAnchorIndex], anchor_dict[anchor_index])
			x = minDist
			c = DistFunc(pixel, anchor_dict[anchor_index])
			if(math.pow(c, 2)<=math.pow(x, 2) + 4*delta*eps):
				partition_dict[anchor_index].append(pixel)'''

def check_equal(pixel1, pixel2):
	#print("pixel1[0] "+str(pixel1[0]))
	#print("pixel2[0] "+str(pixel2[0]))
	if(pixel1[0]==pixel2[0] and pixel1[1]==pixel2[1] and pixel1[2]==pixel2[2]):
		return True
	else: 
		return False


def reduce_function(arg_list):
	global resultReduce, partition_dict_important, partition_dict

	partition_keys = arg_list
	

	#print("Part dict id "+str(partition_dict_id))

	nr_p = os.getpid()
	similarityPixels = list()
	for partition_dict_id in partition_keys:
		pixelsPartition = [p for p in partition_dict[partition_dict_id]]
	pixelsPartition = np.array(pixelsPartition)

	nr_points = int(math.ceil(0.1*len(pixelsPartition)))

	#print("len pixels partition "+str(len(pixelsPartition)))

	#inner_anchors = get_furthest_points(pixelsPartition, nr_points)
	nr_neighbors = 1
	inner_anchors = get_furthest_points(pixelsPartition, nr_points)

	#intermediary_centroid = centroid(pixelsPartition) #bgr

	#pentru fiecare pixel din partitie calculez cei mai apropiati k vecini, apoi centroidul lor

	intermediary_centroids = list()
	intermediary_anchors_partitions = collections.defaultdict(list)

	for pixel_i in pixelsPartition:
		minDist = 99999
		closestAnchor = [0, 0, 0]
		for inner_anchor in inner_anchors:
			dist_i_anchor = DistFunc(inner_anchor, pixel_i)
			if(dist_i_anchor < minDist):
				minDist = dist_i_anchor
				closestAnchor = inner_anchor
		intermediary_anchors_partitions[(closestAnchor[0], closestAnchor[1], closestAnchor[2])].append(pixel_i)

	#print(intermediary_anchors_partitions)
	for idx in intermediary_anchors_partitions:
		intermediary_centroid = centroid(intermediary_anchors_partitions[idx])
		intermediary_centroids.append(intermediary_centroid)


	intermediary_centroids = agglomerative_clustering2(intermediary_centroids, 3)
		

	return intermediary_centroids


def clusterJoinCuloriNaturale(path_to_image, nrClustersParam, grad_detaliere):
	
	global total_anchors, partition_dict, partition_dict_important, resultReduce, nrClusters

	initial_image = cv2.imread(path_to_image)
	initial_image = cv2.cvtColor(initial_image, cv2.COLOR_BGR2RGB)
	image_to_cluster = cv2.imread(path_to_image)
	image_to_cluster = cv2.cvtColor(image_to_cluster, cv2.COLOR_BGR2RGB)

	image_grayscale = cv2.imread(path_to_image, 0)

	nrClusters = nrClustersParam

	#resize image 100 = noua latime

	r = 100.0 / image_to_cluster.shape[1]
	dim = (100, int(image_to_cluster.shape[0] * r))

	image_to_cluster = cv2.resize(image_to_cluster, dim, interpolation = cv2.INTER_AREA)
	image_grayscale = cv2.resize(image_grayscale, dim, interpolation = cv2.INTER_AREA)
	image_pixels = image_to_cluster

	image_width = len(image_to_cluster[0])
	image_height = len(image_to_cluster)

	print("width = "+str(image_width)+" height = "+str(image_height))

	'''change shape to an array with channels as elements '''
	image_to_cluster = np.reshape(image_to_cluster, (image_width*image_height, 3))

	image_width_reshaped = len(initial_image[0])
	image_height_reshaped = len(initial_image)

	initial_image_reshaped = np.reshape(initial_image, (image_width_reshaped*image_height_reshaped, 3))


	no_duplicates = unique_rows(image_to_cluster)

	print np.shape(no_duplicates)

	xs = list(); ys = list(); zs = list();
	xs_anchor = list(); ys_anchor = list(); zs_anchor = list();
	xs_anchor_prob = list(); ys_anchor_prob = list(); zs_anchor_prob = list();
	xs_intermediary = list(); ys_intermediary = list(); zs_intermediary = list();


	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	#ax_intermediary_points = fig.add_subplot(222, projection='3d')

	ax.set_xlim3d(0, 256)
	ax.set_ylim3d(0,256)
	ax.set_zlim3d(0,256)

	'''ax_intermediary_points.set_xlim3d(0, 256)
	ax_intermediary_points.set_ylim3d(0,256)
	ax_intermediary_points.set_zlim3d(0,256)'''

	'''for q in range(0, len(no_duplicates)):
		xs.append(no_duplicates[q][0])
		ys.append(no_duplicates[q][1])
		zs.append(no_duplicates[q][2])


	ax.scatter(xs, ys, zs, c='g', marker='o', depthshade=0)'''
	#chans = cv2.split(image_pixels)
	
	'''color = ('g') #color = ('b','g','r') https://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/ https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
	histr_colors = list()
	pixels_per_bin = 8  #16 binuri, 16 pixeli pe bin (256%nr_binuri = nr_pixeli/bin)
	for i,col in enumerate(color):
		histr = cv2.calcHist([image_pixels],[i],None,[32],[0,256]) #16 -> nrul de bin-uri
		print(np.shape(histr))
		#print(histr) #printare histograma
		#plt.plot(histr,color = col)
		#plt.xlim([0,32])'''
	pixels_per_bin = 8
	#image_pixels = cv2.cvtColor(image_pixels, cv2.COLOR_RGB2GRAY)
	image_pixels = image_grayscale
	histr = cv2.calcHist([image_pixels],[0],None,[32],[0,256])
	

	print("SHAPE >>> "+str(np.shape(image_grayscale)))
	print("nr pixeli >>> "+str(image_width*image_height))

		

	histr = list(histr)
	print("shape histr= "+str(np.shape(histr)))
	print("len histr "+str(len(histr)))
	anchors_histr_fin = list()

	medie_pixeli_per_bin = 0
	suma_pixeli_per_bin = 0
	for bin_id in range(len(histr)):
		suma_pixeli_per_bin = suma_pixeli_per_bin + histr[bin_id]

	print("suma binuri = "+str(suma_pixeli_per_bin))

	for bin_id in range(len(histr)):
		print("["+str(bin_id)+"] = "+str(histr[bin_id]))

	medie_pixeli_per_bin = suma_pixeli_per_bin / len(histr)

	print("medie pixeli per bin = "+str(medie_pixeli_per_bin))

	prag1 = medie_pixeli_per_bin
	prag2 = medie_pixeli_per_bin*0.1

	image_pixels_reshaped = np.reshape(image_pixels, (image_width*image_height, 1)) #e deja grayscale
	print(np.shape(image_pixels_reshaped))
	for bin_id in range(len(histr)):
		#selectez x% pixeli din fiecare bin
		px_number = 0
		if(histr[bin_id]>prag2):
			px_number = int(histr[bin_id]*0.05)
		else:
			print("NR BIN MAI MIC "+str(bin_id))
		
		pixels_in_bin = list()
		
		for id_pixel in range(len(image_pixels_reshaped)):
			if( (int(image_pixels_reshaped[id_pixel]/pixels_per_bin)==bin_id)):
				pixels_in_bin.append(image_to_cluster[id_pixel])
		
		if(len(pixels_in_bin) > 0):
			far_points = get_furthest_points(pixels_in_bin, px_number)
			for far_point in far_points:
				anchors_histr_fin.append(far_point)
		
	partition_dict = collections.defaultdict(list)
	partition_dict_important = collections.defaultdict(list)
	anchor_dict = {}
	for id_anchor, anchor in enumerate(anchors_histr_fin):
		anchor_dict[id_anchor] = anchor
		partition_dict[id_anchor].append(anchor)

	#ploteaza punctele imaginii
	'''for q in range(0, len(image_to_cluster)):
		xs.append(image_to_cluster[q][0])
		ys.append(image_to_cluster[q][1])
		zs.append(image_to_cluster[q][2])

	ax.scatter(xs, ys, zs, c='g', marker='^', depthshade=0)'''


	#ploteaza 3d ancorele
	for q in range(0, len(anchors_histr_fin)):
		xs_anchor_prob.append(anchors_histr_fin[q][0])
		ys_anchor_prob.append(anchors_histr_fin[q][1])
		zs_anchor_prob.append(anchors_histr_fin[q][2])

	ax.scatter(xs_anchor_prob, ys_anchor_prob, zs_anchor_prob, c='r', marker='o', depthshade=0)


	#map
	start_time = time.time()
	args_map = [image_to_cluster, anchor_dict, partition_dict, 1]
	map_function(args_map)
	#print("AM IESIT DIN MAP");
	
	#2dhashing partitii mari
	'''total_anchors = len(partition_dict)
	for partition_dict_id in partition_dict.keys():
		if(len(partition_dict[partition_dict_id])>100):
			#daca am mai mult de 50 de px in partitie incep 2d hashing
			#print("dimensiune partitie "+str(len(partition_dict[partition_dict_id])))
			#print("dimensiune dictionar "+str(len(partition_dict)))
			HashPartition(partition_dict[partition_dict_id], partition_dict)
			#print("cheie de sters "+str(partition_dict_id))
			del partition_dict[partition_dict_id] #stergem elementul deoarece l-am transformat in 4 partitii
			print("dimensiune dictionar dupa hash "+str(len(partition_dict)))
	print("AM IESIT DIN HASHING");
	print("Numar ancore "+str(len(partition_dict)))'''

	suma_dict = 0
	for k in partition_dict:
		print("Dimensiune cluster k= "+str(k)+" "+str(len(partition_dict[k])))
		suma_dict = suma_dict + len(partition_dict[k])
	medie_dict = suma_dict/len(partition_dict)

	print("Medie clustere = "+str(medie_dict))
	prag = medie_dict*grad_detaliere
	print("Prag clustere = "+str(prag))
	for k in partition_dict:
		if len(partition_dict[k]) > prag:
			partition_dict_important[k] = partition_dict[k]


	#ploteaza clusterele

	'''for it in partition_dict_important:
		xs_anchor_prob = list()
		ys_anchor_prob = list()
		zs_anchor_prob = list()
		anchors_partition_dict = partition_dict_important[it] 
		print("DIMENSIUNE = "+str(len(anchors_partition_dict)))
		for q in range(0, len(anchors_partition_dict)):
			xs_anchor_prob.append(anchors_partition_dict[q][0])
			ys_anchor_prob.append(anchors_partition_dict[q][1])
			zs_anchor_prob.append(anchors_partition_dict[q][2])
		colorRandom = random_color_scaled()
		#print colorRandom
		ax.scatter(xs_anchor_prob, ys_anchor_prob, zs_anchor_prob, c=colorRandom, marker='o', depthshade=1)'''

	
	#reduce	
	start_time_r = time.time()
	resultReduce = mp.Manager().dict()

	print("partition dict important len "+str(len(partition_dict_important)))

	data_pairs = list()
	partitions_per_reducer = 4
	start = 0
	end = len(partition_dict_important)
	number_of_pieces = int(math.floor(len(partition_dict_important)/partitions_per_reducer))
	print("no pieces"+str(number_of_pieces))
	partition_dict_important_keys = list(partition_dict_important.keys())
	while_iterator = 0

	while(while_iterator<number_of_pieces):
		
		partition_dict_iterator = start+partitions_per_reducer
		#print("part dict iterator "+str(partition_dict_iterator))
		helper = list()
		for k in range(start, partition_dict_iterator):
			#print("k= "+str(k))
			helper.append(partition_dict_important_keys[k])
		data_pairs.append(helper)
		start = partition_dict_iterator
		while_iterator = while_iterator + 1
	
	print(data_pairs)
	
	#print(data_pairs)
	pool = Pool(30)
	intermediary_centroids_list_of_lists = pool.map(reduce_function,data_pairs) 
	#print("--- %s secunde dureaza REDUCE---" % (time.time() - start_time_r))

	pool.close()
	pool.join()

	intermediary_centroids = list()

	print(np.shape(intermediary_centroids_list_of_lists))

	for list_of_centroids in intermediary_centroids_list_of_lists:
		for centr in list_of_centroids:
			intermediary_centroids.append(centr)

	#print(np.shape(intermediary_centroids))
	#print("========================================")
	#print(intermediary_centroids)


	clusters = agglomerative_clustering2(intermediary_centroids, nrClusters)
	

	#print("--- %s secunde dureaza ALGORITMUL KNN---" % (time.time() - start_time))

	'''xs_intermediary_cluster = list()
	ys_intermediary_cluster = list()
	zs_intermediary_cluster = list()

	for q in range(0, len(clusters)):
		xs_intermediary_cluster.append(clusters[q][0])
		ys_intermediary_cluster.append(clusters[q][1])
		zs_intermediary_cluster.append(clusters[q][2])

	ax.scatter(xs_intermediary_cluster, ys_intermediary_cluster, zs_intermediary_cluster, c='r', marker='o', depthshade=0)
	'''
	plt.show()

	print(clusters)

	#FINALIZARE IMAGINE SI ARANJARE CULORI NATURALE
	colorsResp = list()
	colorsHexaResp = list()
	colorsNrPixels = {}

	for color in clusters:
		colorsNrPixels[(color[0], color[1], color[2])]=0

	for pixel_id in range(len(initial_image_reshaped)):
		
		minDist = 9999
		color = [0,0,0]
		for cluster in clusters:
			distClust = DistFunc(initial_image_reshaped[pixel_id], cluster)
			if(distClust < minDist):
				minDist = distClust
				color = cluster
		initial_image_reshaped[pixel_id][0] = color[0]
		initial_image_reshaped[pixel_id][1] = color[1]
		initial_image_reshaped[pixel_id][2] = color[2]
		colorsNrPixels[(color[0], color[1], color[2])] = colorsNrPixels[(color[0], color[1], color[2])] + 1

	#print("INAINTE")
	print((initial_image_reshaped[10][0],initial_image_reshaped[10][1],initial_image_reshaped[10][2]))
	print((initial_image_reshaped[100][0],initial_image_reshaped[100][1],initial_image_reshaped[100][2]))

	for color in clusters:
		colorsNrPixels[(color[0], color[1], color[2])] = round( (colorsNrPixels[(color[0], color[1], color[2])]/len(initial_image_reshaped)*100), 2)
		colorsResp.append((color[0], color[1], color[2]))
		colorHexa = '#%02x%02x%02x' % (color[0], color[1], color[2])
		colorsHexaResp.append(colorHexa)
	initial_image_reshaped = np.reshape(initial_image_reshaped, (image_height_reshaped, image_width_reshaped, 3))

	cv2.imwrite("/home/iuliar/CERCETARE_INTERFATA/IMAGINI_TEMPORAR/rezultat.png", cv2.cvtColor(initial_image_reshaped, cv2.COLOR_RGB2BGR))

	return (colorsNrPixels, colorsResp, colorsHexaResp)
	
#!!!!!
clusterJoinCuloriNaturale("/home/iuliar/CERCETARE_INTERFATA/IMAGINI_PRELUATE/5.png", 3, 1)