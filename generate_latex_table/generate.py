from __future__ import division

import numpy as np
import sys
import math


filenames = ["rezultate_evaluare_MariaClust.txt", "rezultate_evaluare_BIRCH.txt", "rezultate_evaluare_CLARANS.txt", "rezultate_evaluare_CURE.txt", "rezultate_evaluare_GAUSSIANMIXTURE.txt", "rezultate_evaluare_HIERARCHICALAGG.txt", "rezultate_evaluare_KMEANS.txt", "rezultate_evaluare_OPTICS.txt", "rezultate_evaluare_SPECTRALCLUSTERING.txt"]
names = ["MariaClust", "Birch", "Clarans", "Cure", "Gaussian Mixture", "Hierarchical", "KMeans", "Optics", "Spectral Clustering"]
datasets = ["Aggregation", "Compound", "D31", "Flame", "Jain", "pathbased", "R15", "Spiral", "Dim032", "Dim064", "Dim128", "Dim256", "Dim512"]
file_stats = list()

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
f_latex = open("latexOutput.txt", "a")

f_latex.write("\\begin{table}[!ht]\n")
f_latex.write("\\centering\n")
f_latex.write("\\tiny\n")
f_latex.write("\\caption{Purity evaluation results}\n")
f_latex.write("\\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|}\n")
f_latex.write("\\hline\n")
f_latex.write("\\textbf{Dataset} & \\textbf{MariaClust} & \\textbf{Birch} & \\textbf{Clarans} & \textbf{Cure} & \\textbf{\\begin{tabular}[c]{@{}l@{}}Gaussian\\\\ Mixture\end{tabular}} & \\textbf{Hierarchical} & \textbf{K-Means} & \\textbf{Optics} & \\textbf{\\begin{tabular}[c]{@{}l@{}}Sectral \\\\ Custering\end{tabular}} \\\\ \\hline\n")

for dataset_id in range(len(datasets)):
	table_line = datasets[dataset_id]+"\t"
	for filename_id in range(len(filenames)):
		table_line = table_line+"& "+str(file_stats[filename_id][0][dataset_id])+"\t"
	table_line = table_line+"\\\\ \\hline\n"
	
	f_latex.write(table_line)

f_latex.write("\\end{tabular}\n")
f_latex.write("\\end{table}\n")
f_latex.write("=================================================\n\n")


f_latex.write("\\begin{table}[!ht]\n")
f_latex.write("\\centering\n")
f_latex.write("\\tiny\n")
f_latex.write("\\caption{Entropy evaluation results}\n")
f_latex.write("\\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|}\n")
f_latex.write("\\hline\n")
f_latex.write("\\textbf{Dataset} & \\textbf{MariaClust} & \\textbf{Birch} & \\textbf{Clarans} & \textbf{Cure} & \\textbf{\\begin{tabular}[c]{@{}l@{}}Gaussian\\\\ Mixture\end{tabular}} & \\textbf{Hierarchical} & \textbf{K-Means} & \\textbf{Optics} & \\textbf{\\begin{tabular}[c]{@{}l@{}}Sectral \\\\ Custering\end{tabular}} \\\\ \\hline\n")

for dataset_id in range(len(datasets)):
	table_line = datasets[dataset_id]+"\t"
	for filename_id in range(len(filenames)):
		table_line = table_line+"& "+str(file_stats[filename_id][1][dataset_id])+"\t"
	table_line = table_line+"\\\\ \\hline\n"
	
	f_latex.write(table_line)

f_latex.write("\\end{tabular}\n")
f_latex.write("\\end{table}\n")
f_latex.write("=================================================\n\n")


f_latex.write("\\begin{table}[!ht]\n")
f_latex.write("\\centering\n")
f_latex.write("\\tiny\n")
f_latex.write("\\caption{Rand Index evaluation results}\n")
f_latex.write("\\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|}\n")
f_latex.write("\\hline\n")
f_latex.write("\\textbf{Dataset} & \\textbf{MariaClust} & \\textbf{Birch} & \\textbf{Clarans} & \textbf{Cure} & \\textbf{\\begin{tabular}[c]{@{}l@{}}Gaussian\\\\ Mixture\end{tabular}} & \\textbf{Hierarchical} & \textbf{K-Means} & \\textbf{Optics} & \\textbf{\\begin{tabular}[c]{@{}l@{}}Sectral \\\\ Custering\end{tabular}} \\\\ \\hline\n")

for dataset_id in range(len(datasets)):
	table_line = datasets[dataset_id]+"\t"
	for filename_id in range(len(filenames)):
		table_line = table_line+"& "+str(file_stats[filename_id][2][dataset_id])+"\t"
	table_line = table_line+"\\\\ \\hline\n"
	
	f_latex.write(table_line)

f_latex.write("\\end{tabular}\n")
f_latex.write("\\end{table}\n")
f_latex.write("=================================================\n\n")


f_latex.write("\\begin{table}[!ht]\n")
f_latex.write("\\centering\n")
f_latex.write("\\tiny\n")
f_latex.write("\\caption{Adjusted Rand Index evaluation results}\n")
f_latex.write("\\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|}\n")
f_latex.write("\\hline\n")
f_latex.write("\\textbf{Dataset} & \\textbf{MariaClust} & \\textbf{Birch} & \\textbf{Clarans} & \textbf{Cure} & \\textbf{\\begin{tabular}[c]{@{}l@{}}Gaussian\\\\ Mixture\end{tabular}} & \\textbf{Hierarchical} & \textbf{K-Means} & \\textbf{Optics} & \\textbf{\\begin{tabular}[c]{@{}l@{}}Sectral \\\\ Custering\end{tabular}} \\\\ \\hline\n")

for dataset_id in range(len(datasets)):
	table_line = datasets[dataset_id]+"\t"
	for filename_id in range(len(filenames)):
		table_line = table_line+"& "+str(file_stats[filename_id][3][dataset_id])+"\t"
	table_line = table_line+"\\\\ \\hline\n"
	
	f_latex.write(table_line)

f_latex.write("\\end{tabular}\n")
f_latex.write("\\end{table}\n")
f_latex.write("=================================================\n\n")