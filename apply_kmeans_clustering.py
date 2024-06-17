import os
import glob
import pandas as pd
import numpy as np
import umap.umap_ as umap
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score
from clustering_utils import get_random_samples, plot_audio_segments, statistical_report, create_statistical_report_with_radar_plots
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Define the file paths
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset'
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_kmeans_clustering_'

# Create the results directory if it doesn't exist
if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)

# Get a list of all CSV files in the directory
list_files = glob.glob(os.path.join(features_path, '*.csv'))

# Read and concatenate all CSV files into a single dataframe
all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)

# Save the concatenated dataframe with unique call_id to a CSV file
all_data.to_csv(os.path.join(clusterings_results_path, 'all_data.csv'), index=False)

# Drop NaN values
all_data = all_data.dropna()

# Scale data with StandardScaler on used features only
scaler = StandardScaler()
features = all_data.drop(['recording', 'Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)

best_n_clusters = 5
# Perform K-Means clustering with the determined number of clusters
kmeans = KMeans(n_clusters=best_n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(features_scaled)

cluster_membership = kmeans.labels_

# Add cluster membership to the dataframe
all_data['cluster_membership'] = cluster_membership

# Save the results
all_data.to_csv(os.path.join(clusterings_results_path, f'kmeans_cluster_{best_n_clusters}_membership.csv'), index=False)

# # Perform UMAP for obtaining the standard embeddings for the clustering techniques
# umap_reducer = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7)
# standard_embedding = umap_reducer.fit_transform(features_scaled)


# # Transform the KMeans cluster centers using the UMAP reducer
# umap_centroids = umap_reducer.transform(kmeans.cluster_centers_)

# print("Original Data Shape:", features_scaled.shape)
# print("UMAP Embedding Shape:", standard_embedding.shape)
# print("KMeans Cluster Centers Shape:", kmeans.cluster_centers_.shape)
# print("Transformed UMAP Centroids Shape:", umap_centroids.shape)




# colors = ['lightgreen', 'lightskyblue', 'lightpink', 'navajowhite', 'lightseagreen', 'lightcoral', 'lightgrey', 'lightyellow', 'lightblue', 'lightgreen']
# # Plot the UMAP embedding with the cluster membership in 3 dimensions
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for i in range(best_n_clusters):
#     ax.scatter(standard_embedding[cluster_membership == i, 0], standard_embedding[cluster_membership == i, 1],
#                standard_embedding[cluster_membership == i, 2], c=colors[i], label=f'Cluster {i}', alpha=0.1)

# # Plot the cluster centers
# ax.scatter(umap_centroids[:, 0], umap_centroids[:, 1], umap_centroids[:, 2], color='crimson', marker='x', s=80, label='Centroids')
# for j in range(best_n_clusters):
#     ax.text(umap_centroids[j, 0], umap_centroids[j, 1], umap_centroids[j, 2], str(j+1), color='k', fontsize=10, fontweight='bold')

# ax.set_title(f'KMeans Clustering with {best_n_clusters} clusters')
# ax.set_xlabel('UMAP 1')
# ax.set_ylabel('UMAP 2')
# ax.set_zlabel('UMAP 3')
# ax.legend()
# plt.savefig(os.path.join(clusterings_results_path, f'umap_embedding_{best_n_clusters}_clusters.png'))
# plt.show()



# # Get random samples
# random_samples = get_random_samples(all_data, 'cluster_membership', num_samples=5)
# print(' Random samples selected')
# # # Plot the audio segments
# plot_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')

# print('KMeans clustering completed')

# stats = statistical_report(all_data, cluster_membership,best_n_clusters, metadata, clusterings_results_path)


radar_results= create_statistical_report_with_radar_plots(all_data, cluster_membership, best_n_clusters, metadata, clusterings_results_path)

# print(stats)
print(radar_results)