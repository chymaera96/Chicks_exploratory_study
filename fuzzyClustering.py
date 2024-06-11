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
# from scipy.cluster.hierarchy import dendrogram, linkage

# features_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/_results_high_quality_dataset_'
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_'

# features_path=r'C:\Users\anton\Chicks_Onset_Detection_project\Results_features\_results_examples_'
# metadata_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/High_quality_dataset/high_quality_dataset_metadata.csv'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'


# Path to save the results
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_fuzzy_clustering_'
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


# Clustering:  
# Number of clusters is given by the elbow method on WCSS, silhouette score, calinski_harabasz_score
# except for the DBSCAN methood wich does not require a number of clusters


# 1) Fuzzy clustering
n_max_clusters = 11


fuzzy_cluster_evaluation_per_number_clusters = {
    n_clusters:{'silhouette_score': 0, 
                'calinski_harabasz_score': 0, 
                'wcss':9999, 'fpc':0
                } for n_clusters in range(2, n_max_clusters)
                }




for n_clusters in range(2, n_max_clusters):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(features_scaled.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)
    cluster_membership = np.argmax(u, axis=0)

    # Corrected WCSS computation
    wcss = 0
    for i in range(n_clusters):
        distances = np.linalg.norm(features_scaled - cntr[i], axis=1)
        wcss += np.sum((u[i, :] ** 2) * (distances ** 2))
        
    fuzzy_cluster_evaluation_per_number_clusters[n_clusters]['silhouette_score'] = silhouette_score(features_scaled, cluster_membership)
    fuzzy_cluster_evaluation_per_number_clusters[n_clusters]['calinski_harabasz_score'] = calinski_harabasz_score(features_scaled, cluster_membership)
    fuzzy_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] = wcss  
    fuzzy_cluster_evaluation_per_number_clusters[n_clusters]['fpc'] = fpc
    print('number of clusters:', n_clusters, 'silhouette_score:', fuzzy_cluster_evaluation_per_number_clusters[n_clusters]['silhouette_score'], 'calinski_harabasz_score:', fuzzy_cluster_evaluation_per_number_clusters[n_clusters]['calinski_harabasz_score'], 'wcss:', fuzzy_cluster_evaluation_per_number_clusters[n_clusters]['wcss'], 'fpc:', fuzzy_cluster_evaluation_per_number_clusters[n_clusters]['fpc'])



# save results in the Results_Clustering folder
fuzzy_cluster_evaluation_per_number_clusters_df = pd.DataFrame(fuzzy_cluster_evaluation_per_number_clusters).T


fuzzy_cluster_evaluation_per_number_clusters_df.to_csv(os.path.join(clusterings_results_path, 'fuzzy_cluster_evaluation_per_number_clusters.csv'))
fuzzy_cluster_evaluation_per_number_clusters_df.to_latex(os.path.join(clusterings_results_path, 'fuzzy_cluster_evaluation_per_number_clusters.tex'))

# Find the optimal number of clusters
wcss_vector_across_n_clusters = [fuzzy_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] for n_clusters in range(2, n_max_clusters)]

# best_n_clusters_by_wcss_elbow_finder = find_elbow_point(wcss_vector_across_n_clusters)
wcss_elbow = KneeLocator(range(2, n_max_clusters), wcss_vector_across_n_clusters, curve='convex', direction='decreasing', interp_method='polynomial', online=False)
                
best_n_clusters_by_wcss_elbow = wcss_elbow.elbow    

optimal_k = OptimalK(parallel_backend='joblib')
n_clusters_optimal_k = optimal_k(features_scaled, cluster_array=np.arange(2, n_max_clusters))
print('Optimal number of clusters for k:', n_clusters_optimal_k)
print('Best number of clusters by WCSS elbow:', best_n_clusters_by_wcss_elbow)
plt.figure(figsize=(30, 5))

# First figure
plt.subplot(1, 2, 1)
plt.plot(list(fuzzy_cluster_evaluation_per_number_clusters.keys()), [fuzzy_cluster_evaluation_per_number_clusters[n_clusters]['silhouette_score'] for n_clusters in fuzzy_cluster_evaluation_per_number_clusters.keys()], 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score per number of clusters')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(list(fuzzy_cluster_evaluation_per_number_clusters.keys()), [fuzzy_cluster_evaluation_per_number_clusters[n_clusters]['calinski_harabasz_score'] for n_clusters in fuzzy_cluster_evaluation_per_number_clusters.keys()], 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Calinski Harabasz score')
plt.title('Calinski Harabasz score per number of clusters')
plt.grid()

plt.savefig('C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\fuzzy_cluster_evaluation_per_number_clusters_1.png')

plt.figure(figsize=(30, 5))

# Second figure
plt.subplot(1, 2, 1)
plt.plot(list(fuzzy_cluster_evaluation_per_number_clusters.keys()), [fuzzy_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] for n_clusters in fuzzy_cluster_evaluation_per_number_clusters.keys()], 'bx-')
plt.axvline(x=best_n_clusters_by_wcss_elbow, color='r', linestyle='--', label=f'Elbow point: {best_n_clusters_by_wcss_elbow}')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.legend()
plt.title('WCSS per number of clusters')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(list(fuzzy_cluster_evaluation_per_number_clusters.keys()), [fuzzy_cluster_evaluation_per_number_clusters[n_clusters]['fpc'] for n_clusters in fuzzy_cluster_evaluation_per_number_clusters.keys()], 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('FPC')
plt.legend()
plt.title('FPC per number of clusters')
plt.grid()

plt.savefig('C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\fuzzy_cluster_evaluation_per_number_clusters_2.png')

plt.show()

print('Fuzzy clustering done')
