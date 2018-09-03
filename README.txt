mariaClust_final.py, ultimele modificari
	- am inlocuit pragul de expansiune ca fiind o constanta c*(media distantelor dintre fiecare punct si cei mai apropiati 3 vecini ai lui)
	Imbunatatiri:
	- rapiditate - mult mai rapid pentru Aggregation
	- functioneaza si pentru pathbased cu acuratete mare

Legenda Parametri:
Factor medie - linia 416
Nr. binuri - linia 603
Nr. clustere - linia 761
------------------------------------------------------

Parametri pentru Aggregation:

Metoda calcul clusterizare ierarhica: calculate_centroid (centroid-linkage)

Factor medie: 2
Nr. binuri: 8
Nr. clustere: 7

Parametri pentru jain:

Metoda calcul clusterizare ierarhica: calculate_average_pairwise (average-linkage)

Factor medie: 0.8
Nr. binuri: 3
Nr. clustere: 2

Parametri pentru spiral:

Metoda calcul clusterizare ierarhica: calculate_smallest_pairwise (single-linkage)

Factor medie: 1.8
Nr. binuri: 3
Nr. clustere: 3

Parametri pentru pathbased:
Metoda calcul clusterizare ierarhica: calculate_average_pairwise (average-linkage)

Factor medie: 1.8
Nr. binuri: 2
Nr. clustere: 3

-------------------------------------------------------------------------------------------------
Datasets:

Aggregation: N=788, k=7, D=2 
A. Gionis, H. Mannila, and P. Tsaparas, Clustering aggregation. ACM Transactions on Knowledge Discovery from Data (TKDD), 2007. 1(1): p. 1-30.

Compound: N=399, k=6, D=2 
C.T. Zahn, Graph-theoretical methods for detecting and describing gestalt clusters. IEEE Transactions on Computers, 1971. 100(1): p. 68-86. 

Pathbased: N=300, k=3, D=2 
H. Chang and D.Y. Yeung, Robust path-based spectral clustering. Pattern Recognition, 2008. 41(1): p. 191-203. 

Spiral: N=312, k=3, D=2 
H. Chang and D.Y. Yeung, Robust path-based spectral clustering. Pattern Recognition, 2008. 41(1): p. 191-203. 

D31: N=3100, k=31, D=2 
C.J. Veenman, M.J.T. Reinders, and E. Backer, A maximum variance cluster algorithm. IEEE Trans. Pattern Analysis and Machine Intelligence 2002. 24(9): p. 1273-1280. 

R15: N=600, k=15, D=2 
C.J. Veenman, M.J.T. Reinders, and E. Backer, A maximum variance cluster algorithm. IEEE Trans. Pattern Analysis and Machine Intelligence, 2002. 24(9): p. 1273-1280. 

Jain: N=373, k=2, D=2 
A. Jain and M. Law, Data clustering: A user's dilemma. Lecture Notes in Computer Science, 2005. 3776: p. 1-10. 

Flame: N=240, k=2, D=2 
L. Fu and E. Medico, FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data. BMC bioinformatics, 2007. 8(1): p. 3. 