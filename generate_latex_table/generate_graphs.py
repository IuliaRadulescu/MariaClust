from __future__ import division

import numpy as np
import sys
import math


filenames = ["rezultate_evaluare_MariaClust.txt", "rezultate_evaluare_BIRCH.txt", "rezultate_evaluare_CLARANS.txt", "rezultate_evaluare_CURE.txt", "rezultate_evaluare_GAUSSIANMIXTURE.txt", "rezultate_evaluare_HIERARCHICALAGG.txt", "rezultate_evaluare_KMEANS.txt", "rezultate_evaluare_DBSCAN.txt", "rezultate_evaluare_SPECTRALCLUSTERING.txt"]
names = ["MariaClust", "Birch", "Clarans", "Cure", "Gaussian Mixture", "Hierarchical", "KMeans", "DBSCAN", "Spectral Clustering"]
datasets = ["Aggregation", "Compound", "D31", "Flame", "Jain", "Pathbased", "R15", "Spiral", "Dim032", "Dim064", "Dim128", "Dim256", "Dim512"]
file_stats = list()


def draw_graph(evaluation_values, names, nr, measure):

	colors = ["blue", "red", "green", "orange", "gray", "cyan", "brown", "magenta", "yellow"]
	datasets = ["Aggregation", "Compound", "D31", "Flame", "Jain", "Pathbased", "R15", "Spiral", "Dim032", "Dim064", "Dim128", "Dim256", "Dim512"]

	legend = ','.join(names)
	graph_def = "\t\t\\begin{tikzpicture}[]\n"
	graph_def = graph_def + "\t\t\t\\begin{axis}[\n"
	graph_def = graph_def + "\t\t\t\txmin=0,\n"
	graph_def = graph_def + "\t\t\t\txmax=2,\n"
	graph_def = graph_def + "\t\t\t\txtick,\n"
	graph_def = graph_def + "\t\t\t\tymin=0,\n"
	graph_def = graph_def + "\t\t\t\tymax=1,\n"
	graph_def = graph_def + "\t\t\t\tbar width=4pt,\n"
	graph_def = graph_def + "\t\t\t\tybar=4pt,\n"
	if(nr==0):
		graph_def = graph_def + "\t\t\t\tlegend style={at={(3,0.5)}, anchor=east, legend columns=2}\n\t\t\t\t]\n"
	else:
		graph_def = graph_def + "\t\t\t\t]\n"
	for id_ev_value in range(len(evaluation_values)):
		graph_def = graph_def + "\t\t\t\t\\addplot+[color="+colors[id_ev_value]+"] coordinates {(1,"+str(evaluation_values[id_ev_value])+")};%"+names[id_ev_value]+"\n"
	if(nr==0):
		graph_def = graph_def + "\legend{"+legend+"};\n"
	graph_def = graph_def + "\t\t\t\end{axis}\n"
	graph_def = graph_def + "\t\t\end{tikzpicture}\n"
	graph_def = graph_def + "\t\t\caption{"+datasets[nr]+"}\n"
	caption_measure = measure[0].lower() + measure[1:]
	caption_dataset = datasets[nr][0].lower() + datasets[nr][1:]
	graph_def = graph_def + "\t\t\label{fig:"+caption_measure+"_"+caption_dataset+"}\n"

	return graph_def

for filename_id in range(len(filenames)):

	filename = filenames[filename_id]
	with open(filename) as f:
			content = f.readlines()
	content = [l.strip() for l in content]

	purity = list()
	entropy = list()
	ri = list()
	ari = list()
	for line in content:
		if(line.startswith("Purity")):
			purity_helper = round(float(line.split(":")[1]),3)
			purity.append(purity_helper)
		elif(line.startswith("Entropy")):
			entropy_helper = round(float(line.split(":")[1]),3)
			entropy.append(entropy_helper)
		elif(line.startswith("RI")):
			ri_helper = round(float(line.split(":")[1]),3)
			ri.append(ri_helper)
		elif(line.startswith("ARI")):
			ari_helper = round(float(line.split(":")[1]),3)
			ari.append(ari_helper)
	
	file_stats.append([purity, entropy, ri, ari]) #purity, entropy ri si ari contin pt fiecare set de date
print(np.shape(file_stats))
f_latex = open("latexOutputGraphs.txt", "a")

f_latex.write("\\begin{figure*}[!ht]\n\t\\centering\n")

f_latex.write("\t\\begin{subfigure}{\\columnwidth}\n\t\t\\centering\n")
purity_values = []
for filename_id in range(len(filenames)):
	purity_values.append(file_stats[filename_id][3][0]) #aici modific masura
graph_def = draw_graph(purity_values, names, 0, "ARI")
f_latex.write("\t\t"+graph_def+"\n")
f_latex.write("\t\\end{subfigure}\n")

for dataset_id in range(1, len(datasets)):
	f_latex.write("\t\\begin{subfigure}{0.32\\columnwidth}\n\t\t\\centering\n")
	purity_values = []
	for filename_id in range(len(filenames)):
		purity_values.append(file_stats[filename_id][3][dataset_id]) #aici modific masura
	graph_def = draw_graph(purity_values, names, dataset_id, "ARI")
	f_latex.write(graph_def+"\n")
	f_latex.write("\t\\end{subfigure}\n")
f_latex.write("\t\\caption{ARI comparison}\n")
f_latex.write("\t\\label{fig:ari}\n")
f_latex.write("\\end{figure*}")