
import os
import glob
import pandas as pd
import numpy as np
import umap.umap_ as umap
# import umap  #install umap-learn
import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN, AgglomerativeClustering
from kneed import KneeLocator
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score
from sklearn.metrics import calinski_harabasz_score
# from scipy.cluster.hierarchy import dendrogram, linkage
from clustering_utils import get_random_samples, plot_audio_segments, statistical_report, create_statistical_report_with_radar_plots

# Define the file paths
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset'
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_fuzzy_clustering_'

# Create the results directory if it doesn't exist
if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)

# Get a list of all CSV files in the directory
list_files = glob.glob(os.path.join(features_path, '*.csv'))

# create the name the 'recording' column in the dataframe as the name of the file less features_data  + chicken_id and remove .csv  
# for f in list_files:
#     df = pd.read_csv(f)
#     df['recording'] = os.path.basename(f).replace('features_data_', '').replace('.csv', '')
#     df.to_csv(f, index=False)

# # create the name the 'call_id' column in the dataframe
# for f in list_files:
#     df = pd.read_csv(f)
#     df['call_id'] = df['recording'] + '_' + df['Call Number'].astype(str)
#     df.to_csv(f, index=False)

# print('Data loaded')  

# Read and concatenate all CSV files into a single dataframe
all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)


# Save the concatenated dataframe with unique call_id to a CSV file
all_data.to_csv(os.path.join(clusterings_results_path, 'all_data.csv'), index=False)

# Drop NaN values
all_data = all_data.dropna()

# scale data with StandardScaler on used features only
scaler = StandardScaler()
features = all_data.drop(['recording','Call Number', 'onsets_sec', 'offsets_sec','call_id'], axis=1)
features_scaled = scaler.fit_transform(features)


n_clusters = 4  # best number of clusters from the elbow rule with wcss

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(features_scaled.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)

cluster_membership = np.argmax(u, axis=0)

# all_data['fuzzy_cluster_3_membership'] = cluster_membership


all_data['cluster_membership'] = cluster_membership


# save the results
all_data.to_csv(os.path.join(clusterings_results_path, f'fuzzy_cluster_{n_clusters}_membership.csv'), index=False)


# Perform UMAP for obtaining the standard embeddings for the clustering techniques
umap_reducer = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7)
standard_embedding = umap_reducer.fit_transform(features_scaled)

umap_centroids = []

# Calcolo dei centroidi UMAP
umap_centroids = np.dot(u, standard_embedding) / np.sum(u, axis=1)[:, None]

# Stampa delle dimensioni per verifica (opzionale)
print(f"Dimensioni features_scaled: {features_scaled.shape}")
print(f"Dimensioni standard_embedding: {standard_embedding.shape}")
print(f"Dimensioni u: {u.shape}")
print(f"Dimensioni umap_centroids: {umap_centroids.shape}")



# # Initialize the plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # # Define custom colors for the clusters
# custom_colors = ["darkorange", "turquoise", 'lightGreen', 'LightCoral', 'MediumSlateBlue', 'pink', 'tial', 'yellow', 'DarkOrange', 'DarkGreen', 'DarkBlue', 'DarkViolet']



# # Plot data points with their cluster memberships
# for j in range(n_clusters):
#     ax.scatter(standard_embedding[cluster_membership == j, 0], 
#                standard_embedding[cluster_membership == j, 1], 
#                standard_embedding[cluster_membership == j, 2],
#                color=custom_colors[j % len(custom_colors)], alpha=0.1, label=f'Cluster {j+1}', s=7)

# # Plot the cluster centers
# ax.scatter(umap_centroids[:, 0], umap_centroids[:, 1], umap_centroids[:, 2], color='crimson', marker='x', s=80, label='Centroids')
# for j in range(n_clusters):
#     ax.text(umap_centroids[j, 0], umap_centroids[j, 1], umap_centroids[j, 2], str(j+1), color='k', fontsize=12, fontweight='bold')

# ax.set_title(f'Fuzzy Clustering with {n_clusters} Clusters; FPC = {fpc:.2f}')
# ax.set_xlabel('UMAP 1')
# ax.set_ylabel('UMAP 2')
# ax.set_zlabel('UMAP 3')

# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(clusterings_results_path, f'fuzzy_cluster_{n_clusters}_membership_4d.png'))
# plt.show()


# # Initialize the 2d plot
# fig = plt.figure()
# ax = fig.add_subplot(111)

# # Plot data points with their cluster membership
# for j in range(n_clusters):
#     ax.scatter(standard_embedding[cluster_membership == j, 0], 
#                standard_embedding[cluster_membership == j, 1], 
#                color=custom_colors[j % len(custom_colors)], alpha=0.1, label=f'Cluster {j+1}', s=10)

# # Plot the cluster centers
# ax.scatter(umap_centroids[:, 0], umap_centroids[:, 1], color='r', marker='x', s=100, label='Centroids')
# for j in range(n_clusters):
#     ax.text(umap_centroids[j, 0], umap_centroids[j, 1], str(j+1), color='k', fontsize=12, fontweight='bold')

# ax.set_title(f'Fuzzy Clustering with {n_clusters} Clusters; FPC = {fpc:.2f}')
# ax.set_xlabel('UMAP 1')
# ax.set_ylabel('UMAP 2')

# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(clusterings_results_path, f'fuzzy_cluster_{n_clusters}_membership_2d.png'))
# plt.show()


# # Get random samples
# random_samples = get_random_samples(all_data, 'cluster_membership', num_samples=5)
# print(' Random samples selected')
# # # Plot the audio segments
# plot_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')

# print('Fuzzy clustering completed')

# stats = statistical_report(all_data, cluster_membership,n_clusters, metadata, clusterings_results_path)
# print(stats)


radar_results= create_statistical_report_with_radar_plots(all_data, cluster_membership, n_clusters, metadata, clusterings_results_path)


