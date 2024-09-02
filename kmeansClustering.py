import os
import glob
import pandas as pd
import numpy as np
import umap.umap_ as umap
from gap_statistic import OptimalK
# import umap  #install umap-learn
import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN, AgglomerativeClustering
from kneed import KneeLocator
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score
from clustering_utils import find_elbow_point
from sklearn.cluster import KMeans
# from scipy.cluster.hierarchy import dendrogram, linkage

# features_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/_results_high_quality_dataset_'
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_without_jtfs'

# features_path=r'C:\Users\anton\Chicks_Onset_Detection_project\Results_features\_results_examples_'
# metadata_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/High_quality_dataset/high_quality_dataset_metadata.csv'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'


# Path to save the results
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_kmeans_clustering_without_jtfs'
# Check if the directory exists, if not, create it
if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)

# Get a list of all CSV files in the directory
list_files = glob.glob(os.path.join(features_path, '*.csv'))




all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)

# Drop NaN values
all_data = all_data.dropna()

# scale data with StandardScaler on used features only
scaler = StandardScaler()  
features = all_data.drop(['recording','Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)


# Numero massimo di cluster
n_max_clusters = 11

# Dizionario per memorizzare le valutazioni dei cluster per ogni numero di cluster
kmeans_cluster_evaluation_per_number_clusters = {
    n_clusters: {'silhouette_score': 0,
                 'calinski_harabasz_score': 0,
                 'wcss': 0
                 } for n_clusters in range(2, n_max_clusters)
}

# Iterazione su diverse quantità di cluster
for n_clusters in range(2, n_max_clusters):
    # Creazione di un modello KMeans con n_clusters cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # Addestramento del modello sui dati scalati
    kmeans.fit(features_scaled)
    # Predizione delle etichette dei cluster per ciascun campione
    cluster_membership = kmeans.labels_
    
    # Valutazione del clustering
    silhouette = silhouette_score(features_scaled, cluster_membership)
    calinski_harabasz = calinski_harabasz_score(features_scaled, cluster_membership)
    wcss = kmeans.inertia_
    
    # Memorizzazione delle valutazioni
    kmeans_cluster_evaluation_per_number_clusters[n_clusters]['silhouette_score'] = silhouette
    kmeans_cluster_evaluation_per_number_clusters[n_clusters]['calinski_harabasz_score'] = calinski_harabasz
    kmeans_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] = wcss
    
    print('Numero di cluster:', n_clusters, 'Silhouette Score:', silhouette, 'Calinski Harabasz Score:', calinski_harabasz, 'WCSS:', wcss)

# Salvataggio dei risultati
kmeans_cluster_evaluation_per_number_clusters_df = pd.DataFrame(kmeans_cluster_evaluation_per_number_clusters).T
kmeans_cluster_evaluation_per_number_clusters_df.to_csv(os.path.join(clusterings_results_path, 'kmeans_cluster_evaluation_per_number_clusters.csv'))
kmeans_cluster_evaluation_per_number_clusters_df.to_latex(os.path.join(clusterings_results_path, 'kmeans_cluster_evaluation_per_number_clusters.tex'))

# Trovare il numero ottimale di cluster
wcss_vector_across_n_clusters = [kmeans_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] for n_clusters in range(2, n_max_clusters)]
wcss_elbow = KneeLocator(range(2, n_max_clusters), wcss_vector_across_n_clusters, curve='convex', direction='decreasing', interp_method='polynomial', online=False)
best_n_clusters_by_wcss_elbow = wcss_elbow.elbow

optimal_k = OptimalK(parallel_backend='joblib')
n_clusters_optimal_k = optimal_k(features_scaled, cluster_array=np.arange(2, n_max_clusters))
print('optimal number of clusters according to OptimalK:', n_clusters_optimal_k)
print('best number of clusters according to the elbow rule with WCSS:', best_n_clusters_by_wcss_elbow)

# Grafici
plt.figure(figsize=(30, 5))
plt.subplot(1, 2, 1)
plt.plot(list(kmeans_cluster_evaluation_per_number_clusters.keys()), [kmeans_cluster_evaluation_per_number_clusters[n_clusters]['silhouette_score'] for n_clusters in kmeans_cluster_evaluation_per_number_clusters.keys()], 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette score per number of clusters')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(list(kmeans_cluster_evaluation_per_number_clusters.keys()), [kmeans_cluster_evaluation_per_number_clusters[n_clusters]['calinski_harabasz_score'] for n_clusters in kmeans_cluster_evaluation_per_number_clusters.keys()], 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Calinski Harabasz Score')
plt.title('Calinski Harabasz Score per number of clusters')
plt.grid()

plt.savefig('C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_kmeans_clustering_\\kmeans_cluster_evaluation_per_number_clusters_1.png')

# plot just the WCSS
plt.figure(figsize=(10, 5))
plt.plot(list(kmeans_cluster_evaluation_per_number_clusters.keys()), [kmeans_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] for n_clusters in kmeans_cluster_evaluation_per_number_clusters.keys()], 'bx-')
plt.axvline(x=best_n_clusters_by_wcss_elbow, color='r', linestyle='--', label=f'Elbow point: {best_n_clusters_by_wcss_elbow}')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('WCSS per number of clusters')
plt.legend()


plt.savefig('C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_kmeans_clustering_\\kmeans_cluster_evaluation_per_number_clusters_2.png')

plt.show()

print('Clustering with KMeans completed!')
