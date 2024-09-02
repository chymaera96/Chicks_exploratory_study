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
from clustering_utils import get_random_samples, plot_audio_segments, statistical_report, create_statistical_report_with_radar_plots, plot_and_save_audio_segments, plot_and_save_extreme_calls

# Define the file paths
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\High_quality_dataset'
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_fuzzy_clustering_\\_2_clusters_'

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

# scale data with StandardScaler on used features only
scaler = StandardScaler()
features = all_data.drop(['recording','Call Number', 'onsets_sec', 'offsets_sec','call_id'], axis=1)
features_scaled = scaler.fit_transform(features)


n_clusters = 2 # best number of clusters from the elbow rule with wcss

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



# Initialize the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # Define custom colors for the clusters
custom_colors = ["darkorange", "turquoise", 'lightGreen', 'LightCoral', 'MediumSlateBlue', 'pink', 'tial', 'yellow', 'DarkOrange', 'DarkGreen', 'DarkBlue', 'DarkViolet']



# Plot data points with their cluster memberships
for j in range(n_clusters):
    ax.scatter(standard_embedding[cluster_membership == j, 0], 
               standard_embedding[cluster_membership == j, 1], 
               standard_embedding[cluster_membership == j, 2],
               color=custom_colors[j % len(custom_colors)], alpha=0.1, label=f'Cluster {j+1}', s=7)

# Plot the cluster centers
ax.scatter(umap_centroids[:, 0], umap_centroids[:, 1], umap_centroids[:, 2], color='crimson', marker='x', s=80, label='Centroids')
for j in range(n_clusters):
    ax.text(umap_centroids[j, 0], umap_centroids[j, 1], umap_centroids[j, 2], str(j+1), color='k', fontsize=12, fontweight='bold')

ax.set_title(f'Fuzzy Clustering with {n_clusters} Clusters; FPC = {fpc:.2f}')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(clusterings_results_path, f'fuzzy_cluster_{n_clusters}_membership_4d.png'))
plt.show()


# Initialize the 2d plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Plot data points with their cluster membership
for j in range(n_clusters):
    ax.scatter(standard_embedding[cluster_membership == j, 0], 
               standard_embedding[cluster_membership == j, 1], 
               color=custom_colors[j % len(custom_colors)], alpha=0.1, label=f'Cluster {j+1}', s=10)

# Plot the cluster centers
ax.scatter(umap_centroids[:, 0], umap_centroids[:, 1], color='r', marker='x', s=100, label='Centroids')
for j in range(n_clusters):
    ax.text(umap_centroids[j, 0], umap_centroids[j, 1], str(j+1), color='k', fontsize=12, fontweight='bold')

ax.set_title(f'Fuzzy Clustering with {n_clusters} Clusters; FPC = {fpc:.2f}')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(clusterings_results_path, f'fuzzy_cluster_{n_clusters}_membership_2d.png'))
plt.show()


# Get random samples
random_samples = get_random_samples(all_data, 'cluster_membership', num_samples=5)
print(' Random samples selected')
# # # Plot the audio segments
# plot_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')


#Plot the audio segments and save audio files
plot_and_save_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')

print('Fuzzy clustering completed')

stats = statistical_report(all_data, cluster_membership,n_clusters, metadata, clusterings_results_path)
print(stats)


radar_results= create_statistical_report_with_radar_plots(all_data, cluster_membership, n_clusters, metadata, clusterings_results_path)

# The following code is commented out as it is needed for simple analysis of the fuzzy clustering 
# the aim is to compute the probability of each call to belong to each cluster and select the most rappresentative calls for each cluster
# Save the probability arrays in fuzzy_cluster_{n_clusters}_membership.csv file

# # Add probabilities to the dataframe
# probabilities_df = pd.DataFrame(u.T, columns=['Probability Cluster 0', 'Probability Cluster 1','Probability Cluster 2'])
# all_data = pd.concat([all_data, probabilities_df], axis=1)
# all_data.to_csv(os.path.join(clusterings_results_path, f'fuzzy_cluster_{n_clusters}_membership_probabilities.csv'), index=False)

# # #Save and cluster values to a CSV file
# cluster_0_values = u[0]
# cluster_1_values = u[1]
# cluster_values_df = pd.DataFrame({'cluster_0_values': cluster_0_values, 'cluster_1_values': cluster_1_values})
# cluster_values_df.to_csv(os.path.join(clusterings_results_path, 'fuzzy_clustering_values.csv'), index=False)

# # Create a scatter plot with jitter
# plt.figure(figsize=(10, 8))
# jitter = np.random.normal(0, 0.01, size=cluster_0_values.shape)
# # Plot points for cluster 0
# plt.scatter(cluster_1_values + jitter, cluster_0_values + jitter, c='blue', alpha=0.3, s=10, label='Cluster 0')
# # Plot points for cluster 1
# plt.scatter(cluster_0_values + jitter, cluster_1_values + jitter, c='red', alpha=0.3, s=10, label='Cluster 1')
# # Set labels and title
# plt.xlabel('Probability of belonging to Cluster 1')
# plt.ylabel('Probability of belonging to Cluster 0')
# plt.title('Cluster Membership Probabilities for Fuzzy Clustering')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.savefig(os.path.join(clusterings_results_path, 'cluster_membership_posterior_probabilities.png'))
# plt.show()

# # Extract the top 5 calls for each cluster from each individual recording
# top_calls_cluster_0 = all_data.groupby('recording', group_keys=False).apply(lambda x: x.nlargest(5, 'Probability Cluster 0'))
# top_calls_cluster_0 = top_calls_cluster_0[top_calls_cluster_0['cluster_membership'] == 0]

# top_calls_cluster_1 = all_data.groupby('recording', group_keys=False).apply(lambda x: x.nlargest(5, 'Probability Cluster 1'))
# top_calls_cluster_1 = top_calls_cluster_1[top_calls_cluster_1['cluster_membership'] == 1]

# # Save top calls to CSV
# top_calls_cluster_0.to_csv(os.path.join(clusterings_results_path, 'top_calls_cluster_0.csv'), index=False)
# top_calls_cluster_1.to_csv(os.path.join(clusterings_results_path, 'top_calls_cluster_1.csv'), index=False)

# # Plot and save the top calls
# plot_and_save_extreme_calls(top_calls_cluster_0, audio_path, clusterings_results_path)
# plot_and_save_extreme_calls(top_calls_cluster_1, audio_path, clusterings_results_path)


# Save and cluster values to a CSV file
# cluster_values_df = pd.DataFrame(u.T, columns=[f'cluster_{i}_values' for i in range(n_clusters)])
# cluster_values_df.to_csv(os.path.join(clusterings_results_path, 'fuzzy_clustering_values.csv'), index=False)

# Create a scatter plot with jitter for each cluster pair
# plt.figure(figsize=(10, 8))
# jitter = np.random.normal(0, 0.01, size=u.shape[1])

# for i in range(n_clusters):
#     for j in range(i+1, n_clusters):
#         plt.scatter(u[i] + jitter, u[j] + jitter, alpha=0.3, s=10, label=f'Cluster {i} vs Cluster {j}')

# plt.xlabel('Probability of belonging to Cluster i')
# plt.ylabel('Probability of belonging to Cluster j')
# plt.title('Cluster Membership Probabilities for Fuzzy Clustering')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.savefig(os.path.join(clusterings_results_path, 'cluster_membership_posterior_probabilities.png'))
# plt.show()
# Extract the top 5 calls for each cluster from each individual recording

# top_calls = {f'top_calls_cluster_{i}': all_data.groupby('recording', group_keys=False).apply(lambda x: x.nlargest(5, f'Probability Cluster {i}')) for i in range(n_clusters)}

# for i in range(n_clusters):
#     top_calls[f'top_calls_cluster_{i}'] = top_calls[f'top_calls_cluster_{i}'][top_calls[f'top_calls_cluster_{i}']['cluster_membership'] == i]
#     top_calls[f'top_calls_cluster_{i}'].to_csv(os.path.join(clusterings_results_path, f'top_calls_cluster_{i}.csv'), index=False)
#     plot_and_save_extreme_calls(top_calls[f'top_calls_cluster_{i}'], audio_path, clusterings_results_path)
