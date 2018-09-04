# MariaClus

### Run parameters:
* filename of the dataset
* no_clusters
* no_bins
* expand_factor - given a center, how much a cluster can expand based on the number of neighbours
* how to compute the inter cluster dinstances:
	* 1 = centroid linkage
	* 2 = average linkage
	* 3 = single linkage
	* 4 = average linkage ponderat

Run example: python3 mariaClust.py aggregation.txt 7 8 1 1

-------------------------------------------------------------------------------------------------

## Datasets:

### Aggregation: N=788, k=7, D=2 
* Aggregation: No. Clusters: 7, No. Bins: 8, Expand Factor: 1, Dinstance Type: 1 (centroid)
* python3 src/mariaClust.py datasets/aggregation.txt 7 8 1 1
* A. Gionis, H. Mannila, and P. Tsaparas, Clustering aggregation. ACM Transactions on Knowledge Discovery from Data (TKDD), 2007. 1(1): p. 1-30.

### Spiral: N=312, k=3, D=2 
* Spiral: No. Clusters: 3, No. Bins: 3, Expand Factor: 1.8, Dinstance Type: 3 (single linkage)
* python3 src/mariaClust.py datasets/spiral.txt 3 3 1.8 3
* H. Chang and D.Y. Yeung, Robust path-based spectral clustering. Pattern Recognition, 2008. 41(1): p. 191-203. 

### R15: N=600, k=15, D=2 
* R15: No. Clusters: 15, No. Bins: 8, Expand Factor: 1, Dinstance Type: 1 (centroid)
* python3 src/mariaClust.py datasets/r15.txt 15 8 1 1
* C.J. Veenman, M.J.T. Reinders, and E. Backer, A maximum variance cluster algorithm. IEEE Trans. Pattern Analysis and Machine Intelligence, 2002. 24(9): p. 1273-1280. 

### Jain: N=373, k=2, D=2 
* Jain: No. Clusters: 2, No. Bins: 3, Expand Factor: 0.8, Dinstance Type: 2 (average linkage)
* python3 src/mariaClust.py datasets/jain.txt 2 3 0.8 2
* A. Jain and M. Law, Data clustering: A user's dilemma. Lecture Notes in Computer Science, 2005. 3776: p. 1-10. 

### Pathbased: N=300, k=3, D=2 
* Pathbased: No. Clusters: 3, No. Bins: 2, Expand Factor: 1.8, Dinstance Type: 2 (average linkage)
* python3 src/mariaClust.py datasets/pathbased.txt 3 2 1.8 2
* H. Chang and D.Y. Yeung, Robust path-based spectral clustering. Pattern Recognition, 2008. 41(1): p. 191-203. 

### Flame: N=240, k=2, D=2 
* L. Fu and E. Medico, FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data. BMC bioinformatics, 2007. 8(1): p. 3. 

### D31: N=3100, k=31, D=2 
* C.J. Veenman, M.J.T. Reinders, and E. Backer, A maximum variance cluster algorithm. IEEE Trans. Pattern Analysis and Machine Intelligence 2002. 24(9): p. 1273-1280. 

### Compound: N=399, k=6, D=2 
* C.T. Zahn, Graph-theoretical methods for detecting and describing gestalt clusters. IEEE Transactions on Computers, 1971. 100(1): p. 68-86. 

