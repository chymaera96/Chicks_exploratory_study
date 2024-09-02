import os
import glob
import pandas as pd
import numpy as np
import umap.umap_ as umap
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score
from clustering_utils import get_random_samples, plot_audio_segments, statistical_report, create_statistical_report_with_radar_plots, plot_and_save_audio_segments,get_representative_calls_by_percentile, plot_and_save_extreme_calls
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# Define the file paths
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_without_jtfs'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset'
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_kmeans_clustering_without_jtfs\\features_3_clusters_percentile'

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

best_n_clusters = 3
# Perform K-Means clustering with the determined number of clusters
kmeans = KMeans(n_clusters=best_n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(features_scaled)

cluster_membership = kmeans.labels_

# Add cluster membership to the dataframe
all_data['cluster_membership'] = cluster_membership

# Calculate distances for all points to their respective cluster centroids
distances = cdist(features_scaled, kmeans.cluster_centers_, 'euclidean')

all_data['distance_to_centroid'] = distances[np.arange(len(distances)), cluster_membership]



# Define the percentiles to analyze
percentiles = [5, 10, 15, 20,25, 30,35, 40,45,  50,55, 60,65, 70,75, 80,85, 90, 95, 100]


                # Select the columns of interest for means
features = [
    'Duration_call','F0 Mean', 'F0 Std', 'F0 Skewness', 'F0 Kurtosis', 
    'F0 Bandwidth', 'F0 1st Order Diff', 'F0 Slope', 'F0 Mag Mean', 
    'F1 Mag Mean', 'F2 Mag Mean', 'F1-F0 Ratio', 'F2-F0 Ratio', 
    'Spectral Centroid Mean', 'Spectral Centroid Std', 'RMS Mean', 
    'RMS Std', 'Slope', 'Attack_magnitude', 'Attack_time']

# Normalizza i percentili a intervalli tra 0 e 1
normalized_percentiles = [p / 100 for p in percentiles]

# Usa pd.qcut con il numero di gruppi corretto
all_data['percentile'] = pd.qcut(all_data['distance_to_centroid'], q=len(normalized_percentiles), labels=percentiles)

# export and save the dataframe with the percentiles
all_data.to_csv(os.path.join(clusterings_results_path, 'all_data_with_percentiles.csv'), index=False)

clustered_groups = all_data.groupby(['cluster_membership', 'percentile'])

# # Function to get the top 10 closest and 10 farthest points from the centroid
# def get_top_bottom_points(cluster_points, distances, n=10):
#     sorted_indices = np.argsort(distances)
#     closest_indices = sorted_indices[:n]
#     farthest_indices = sorted_indices[-n:]
#     return closest_indices, farthest_indices

# # Collect representative points
# representative_calls = []

# for i in range(best_n_clusters):
#     cluster_mask = cluster_membership == i
#     cluster_points = features_scaled[cluster_mask]
#     cluster_data = all_data[cluster_mask]
#     cluster_distances = distances[cluster_mask]
    
#     closest_indices, farthest_indices = get_top_bottom_points(cluster_points, cluster_distances)
    
#     for idx_type, indices in [('closest', closest_indices), ('farthest', farthest_indices)]:
#         for idx in indices:
#             point_data = cluster_data.iloc[idx].to_dict()
#             point_data['point_type'] = idx_type
#             representative_calls.append(point_data)

# # Create the final DataFrame and save it
# representative_calls_df = pd.DataFrame(representative_calls)
# representative_calls_df.to_csv(os.path.join(clusterings_results_path, 'kmeans_representative_calls.csv'), index=False)

# Save the results with cluster membership and distances
# all_data.to_csv(os.path.join(clusterings_results_path, 'all_data_with_distances.csv'), index=False)

# print("K-means clustering: Representative calls for each cluster saved.")

# count calls for each cluster
calls_per_cluster = all_data['cluster_membership'].value_counts().sort_index()
print("Calls per cluster:")
print(calls_per_cluster)


# Plot the distribution of distances to the centroid for each cluster in separate plots
# for cluster in range(best_n_clusters):
#     cluster_data = all_data[all_data['cluster_membership'] == cluster]
    
#     plt.figure(figsize=(10, 6))
#     plt.hist(cluster_data['distance_to_centroid'], bins=50, alpha=0.7, color='orange', edgecolor='black')
#     plt.title(f'Distance Distribution for Cluster {cluster}')
#     plt.xlabel('Distance to Centroid')
#     plt.ylabel('Frequency')
#     plt.grid(axis='y')
#     plt.savefig(os.path.join(clusterings_results_path, f'cluster_{cluster}_distance_distribution.png'))
#     plt.close()
    
# print("K-means clustering: Distribution of distances to centroid for each cluster plotted.")

# Loop over each cluster
for cluster in range(best_n_clusters):
    cluster_data = all_data[all_data['cluster_membership'] == cluster]
    percentile_groups = cluster_data.groupby('percentile')

    # Loop over each feature
    for feature in features:
        # Initialize plot
        plt.figure(figsize=(12, 8))

        # Create boxplot
        sns.boxplot(x='percentile', y=feature, data=cluster_data, color='moccasin')  

        # Add jittered points on top of boxplot
        sns.stripplot(x='percentile', y=feature, data=cluster_data, color='black', size=3, jitter=True, alpha=0.7)

        # Compute and annotate average distance for each percentile
        avg_distances = cluster_data.groupby('percentile')['distance_to_centroid'].mean().round(2)
        labels = [f'{percentile} ({avg})' for percentile, avg in zip(percentiles, avg_distances)]
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')

        plt.title(f'Cluster {cluster} - {feature} Distribution Across Percentiles')
        plt.xlabel('Percentile (Avg. Distance)')
        plt.ylabel('Feature Value')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(clusterings_results_path, f'cluster_{cluster}_{feature}_jittered_boxplot.png'))
        plt.close()

print("Jittered boxplots for each feature across different percentiles saved.")
    

print("Hierarchical clustering: Cluster membership and distances to centers saved.")




# Selezione delle chiamate rappresentative per ciascun cluster
for cluster in range(best_n_clusters):
    cluster_mask = all_data['cluster_membership'] == cluster
    cluster_data = all_data[cluster_mask]
    
    # Ordina i dati in base alla distanza dal centroide (dalla più vicina alla più lontana)
    cluster_data = cluster_data.sort_values('distance_to_centroid', ascending=True)
    
    # Estrai le chiamate ai percentili specificati
    calls_at_percentiles = get_representative_calls_by_percentile(cluster_data, percentiles)
    
    for percentile, calls in zip(percentiles, calls_at_percentiles):
        # Percorso di salvataggio per le chiamate rappresentative
        save_path = os.path.join(clusterings_results_path, f'cluster_{cluster}_percentile_{percentile}')
        os.makedirs(save_path, exist_ok=True)
        
        # Salva le chiamate rappresentative
        calls.to_csv(os.path.join(save_path, f'representative_calls_cluster_{cluster}_percentile_{percentile}.csv'), index=False)
        
        # Visualizza e salva i segmenti audio
        plot_and_save_audio_segments(calls, audio_path, save_path, f'cluster_{cluster}_percentile_{percentile}')
        
        print(f"\nChiamate rappresentative per il cluster {cluster} al percentile {percentile}:")
        print(calls[['recording', 'call_id', 'distance_to_centroid', 'cluster_membership']])

print("Selection of calls at specified percentiles completed.")













# # Function to get representative calls
# def get_representative_calls(cluster_data, start_rank, n_calls=25):
#     return cluster_data.iloc[start_rank:start_rank+n_calls]

# # Selection of representative calls for each cluster
# for cluster in range(best_n_clusters):
#     cluster_mask = all_data['cluster_membership'] == cluster
#     cluster_data = all_data[cluster_mask]
#     cluster_data = cluster_data.sort_values('distance_to_centroid', ascending=False)

#     # Save sorted cluster data
#     cluster_data.to_csv(os.path.join(clusterings_results_path, f'kmeans_cluster_{cluster}.csv'), index=False)

#     start_rank = 0
#     threshold_found = False
#     while not threshold_found:
#         # Select the next batch of calls
#         representative_calls = get_representative_calls(cluster_data, start_rank)

#         if representative_calls.empty:
#             print(f"Reached the end of calls in cluster {cluster} without finding a threshold.")
#             break

#         # Save path for the current batch
#         save_path = os.path.join(clusterings_results_path, f'cluster_{cluster}_rank_{start_rank+1}_{start_rank+len(representative_calls)}')
#         os.makedirs(save_path, exist_ok=True)

#         # Save representative calls
#         representative_calls.to_csv(os.path.join(save_path, f'representative_calls_cluster_{cluster}_rank_{start_rank+1}_{start_rank+len(representative_calls)}.csv'), index=False)

#         # Plot and save audio segments
#         plot_and_save_audio_segments(representative_calls, audio_path, save_path, f'cluster_{cluster}')

#         print(f"\nAnalysing Cluster {cluster}, Calls {start_rank+1} to {start_rank+len(representative_calls)}:")
#         print(representative_calls[['recording', 'call_id', 'distance_to_centroid']])
#         print("\nPlease analyse these calls.")
#         response = input("Have you found the threshold in this batch? (yes/no): ")

#         if response.lower() == 'yes':
#             threshold_rank = int(input("Enter the rank number where you found the threshold: "))
#             threshold_distance = cluster_data.iloc[threshold_rank-1]['distance_to_centroid']
#             print(f"Threshold found for cluster {cluster} at rank {threshold_rank}, distance {threshold_distance:.4f}")
#             threshold_found = True
#         else:
#             start_rank += len(representative_calls)

#     # After finding threshold, save it
#     if threshold_found:
#         threshold_path = os.path.join(clusterings_results_path, f'cluster_{cluster}_threshold.txt')
#         with open(threshold_path, 'w') as f:
#             f.write(f"Threshold for cluster {cluster}: rank {threshold_rank}\n")
#             f.write(f"This corresponds to a distance_to_centroid of {threshold_distance:.4f}\n")

# print("K-Means clustering: Thresholds for each cluster determined.")

# # Count calls for each cluster
# calls_per_cluster = all_data['cluster_membership'].value_counts().sort_index()
# print("Calls per cluster:")
# print(calls_per_cluster)

# PART TO EXPORT INFORMATION ABOUT THE CLUSTERING DIVISION AND THE DISTANCES TO THE CENTROIDS



# plot_and_save_extreme_calls(representative_calls_df, audio_path, clusterings_results_path)

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


# # # # Plot the audio segments
# # plot_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')

# plot_and_save_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')

# print('KMeans clustering completed')

# stats = statistical_report(all_data, cluster_membership,best_n_clusters, metadata, clusterings_results_path)


# radar_results= create_statistical_report_with_radar_plots(all_data, cluster_membership, best_n_clusters, metadata, clusterings_results_path)

# print(stats)
# print(radar_results)

