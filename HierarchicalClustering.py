import os
import glob
import pandas as pd
import numpy as np
import umap.umap_ as umap
# import umap  #install umap-learn
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score
from sklearn.metrics import calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from clustering_utils import plot_dendrogram, find_elbow_point

from sklearn.cluster import AgglomerativeClustering


# features_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/_results_high_quality_dataset_'
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_'
# metadata_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/High_quality_dataset/high_quality_dataset_metadata.csv'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
# save the results
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering\\_hierarchical_clustering_'
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
features = all_data.drop(['recording', 'Call Number', 'onsets_sec', 'offsets_sec'], axis=1)
features_scaled = scaler.fit_transform(features)


#########################
# # hierarchical clustering without defining the n_clusters- define min distance
distance_thresholds = [50, 75, 100, 125]
cluster_memberships = []
centroids_list = []
n_clusters_list = []

for dt in distance_thresholds:
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=dt, linkage='ward')
    cluster_membership = agg.fit_predict(features_scaled)
    n_clusters = agg.n_clusters_
    centroids = np.array([features_scaled[cluster_membership == c].mean(axis=0) for c in range(n_clusters)])
    
    cluster_memberships.append(cluster_membership)
    centroids_list.append(centroids)
    n_clusters_list.append(n_clusters)
    
    silhouette = silhouette_score(features_scaled, cluster_membership)
    calinski_harabasz = calinski_harabasz_score(features_scaled, cluster_membership)
    wcss = np.sum(np.linalg.norm(features_scaled - centroids[cluster_membership], axis=1) ** 2)
    
    print(f'Distance threshold: {dt}, Number of clusters: {n_clusters}')
    print(f'Silhouette score: {silhouette}')
    print(f'Calinski Harabasz score: {calinski_harabasz}')
    print(f'WCSS: {wcss}\n')

# save results
hierarchical_cluster_evaluation_per_distance_threshold = pd.DataFrame({
    'distance_threshold': distance_thresholds,
    'n_clusters': n_clusters_list,
    'cluster_membership': cluster_memberships,
    'centroids': centroids_list
})
hierarchical_cluster_evaluation_per_distance_threshold.to_csv(os.path.join(clusterings_results_path, 'hierarchical_cluster_evaluation_per_distance_threshold.csv'))



# 2) Hierarchical clustering through the number of clusters
n_max_clusters = 11

hierarchical_cluster_evaluation_per_number_clusters = {
    n_clusters: {'silhouette_score': 0, 
                 'calinski_harabasz_score': 0, 
                 'wcss':9999   
                 } 
                for n_clusters in range(2, n_max_clusters)
                }

for n_clusters in range(2, n_max_clusters):
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward' )

    cluster_membership = agg.fit_predict(features_scaled)
    print('n_clusters:', n_clusters)
    print('n_leaves:', agg.n_leaves_)
    print('n_clusters:', agg.n_clusters_)
    print('n_children:', agg.children_)
    print('n_connected_components:', agg.n_connected_components_)   # number of connected components present or the number of data sets that are connected to each other via a sequence of similar points. 
    
    centroids = np.array([features_scaled[cluster_membership == c].mean(axis=0) for c in range(n_clusters)])

        
    hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['silhouette_score'] = silhouette_score(features_scaled, cluster_membership)
    hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['calinski_harabasz_score'] = calinski_harabasz_score(features_scaled, cluster_membership)
    hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] = np.sum((np.linalg.norm(features_scaled - centroids[cluster_membership], axis=1) ** 2))
  
    print('number of clusters:', n_clusters, 'silhouette_score:', hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['silhouette_score'], 'calinski_harabasz_score:', hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['calinski_harabasz_score'], 'wcss:', hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['wcss'])



# save results
hierarchical_cluster_evaluation_per_number_clusters_df = pd.DataFrame(hierarchical_cluster_evaluation_per_number_clusters).T
hierarchical_cluster_evaluation_per_number_clusters_df.to_csv(os.path.join(clusterings_results_path, 'hierarchical_cluster_evaluation_per_number_clusters.csv'))
# convert to latex and save
hierarchical_cluster_evaluation_per_number_clusters_df.to_latex(os.path.join(clusterings_results_path, 'hierarchical_cluster_evaluation_per_number_clusters.tex'))


# Find the optimal number of clusters
wcss_vector_across_n_clusters = [hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] for n_clusters in range(2, n_max_clusters)]
wcss_elbow = KneeLocator(range(2, n_max_clusters), wcss_vector_across_n_clusters, curve='convex', direction='decreasing', interp_method='polynomial', online=False)
best_n_clusters_by_wcss_elbow= wcss_elbow.elbow



# Imposta la dimensione della figura
fig, axes = plt.subplots(1, 2, figsize=(20, 5))

# Plot per Silhouette score
axes[0].plot(list(hierarchical_cluster_evaluation_per_number_clusters.keys()),[hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['silhouette_score'] for n_clusters in hierarchical_cluster_evaluation_per_number_clusters.keys()],'bx-')
axes[0].set_xlabel('Number of clusters')
axes[0].set_ylabel('Silhouette score')
axes[0].set_title('Silhouette score per number of clusters')
axes[0].grid(True)

# Plot per Calinski Harabasz score
axes[1].plot(list(hierarchical_cluster_evaluation_per_number_clusters.keys()),[hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['calinski_harabasz_score'] for n_clusters in hierarchical_cluster_evaluation_per_number_clusters.keys()],'bx-')
axes[1].set_xlabel('Number of clusters')
axes[1].set_ylabel('Calinski Harabasz score')
axes[1].set_title('Calinski Harabasz score per number of clusters')
axes[1].grid(True)
plt.tight_layout()
plt.savefig('C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering\\_hierarchical_clustering_\\hierarchical_cluster_evaluation_per_number_clusters_silhouette_calinski.png')
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(list(hierarchical_cluster_evaluation_per_number_clusters.keys()),[hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] for n_clusters in hierarchical_cluster_evaluation_per_number_clusters.keys()],'bx-')
plt.axvline(x=best_n_clusters_by_wcss_elbow, color='r', linestyle='--', label=f'Elbow point: {best_n_clusters_by_wcss_elbow}')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('WCSS per number of clusters')
plt.legend()
plt.grid(True)
plt.savefig('C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering\\_hierarchical_clustering_\\hierarchical_cluster_evaluation_per_number_clusters_wcss.png')
plt.show()

print('Hierarchical clustering done')
