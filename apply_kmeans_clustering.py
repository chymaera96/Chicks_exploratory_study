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

# Determine the number of clusters using the elbow method with WCSS
wcss_values = []
for i in range(1, 12):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(features_scaled)
    wcss_values.append(kmeans.inertia_)

# Find the elbow point
knee_locator = KneeLocator(range(1, 12), wcss_values, curve='convex', direction='decreasing')
best_n_clusters = knee_locator.elbow

# Perform K-Means clustering with the determined number of clusters
kmeans = KMeans(n_clusters=best_n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_membership = kmeans.fit_predict(features_scaled)

# Add cluster membership to the dataframe
all_data['cluster_membership'] = cluster_membership

# Save the results
all_data.to_csv(os.path.join(clusterings_results_path, f'kmeans_cluster_{best_n_clusters}_membership.csv'), index=False)

# Perform UMAP for obtaining the standard embeddings for the clustering techniques
umap_reducer = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7)
standard_embedding = umap_reducer.fit_transform(features_scaled)

umap_centroids = np.dot(kmeans.cluster_centers_, umap_reducer.embedding_) / np.sum(kmeans.cluster_centers_, axis=1)[:, None]

# Plot the results and centroids
# (code for plotting omitted for brevity)

# Get statistical report and radar plots
stats = statistical_report(all_data, cluster_membership, best_n_clusters, metadata, clusterings_results_path)
print(stats)

radar_results = create_statistical_report_with_radar_plots(all_data, cluster_membership, best_n_clusters, metadata, clusterings_results_path)
