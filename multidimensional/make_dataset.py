import sys
import os


if __name__ == "__main__":
	filename_dimensions = sys.argv[1]
	filename_truth = sys.argv[2]

	with open(filename_dimensions) as f:
		content_dimensions = f.readlines()

	with open(filename_truth) as f:
		content_truth = f.readlines()

	content_dimensions = [l.strip() for l in content_dimensions]
	content_truth = [l.strip() for l in content_truth]

	file_merged = open("dim032_merged.txt","w") 

	for id_line in range(len(content_dimensions)):
		aux_dimensions = content_dimensions[id_line].split('\t')
		aux_truth = content_truth[id_line].split('\t')
		aux_dimensions = aux_dimensions + aux_truth
		line_combined = '\t'.join(aux_dimensions)
		file_merged.write(line_combined+"\n")
	file_merged.close()
		
