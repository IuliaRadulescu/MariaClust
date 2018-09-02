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