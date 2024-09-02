import os
import glob
import pandas as pd
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from clustering_utils import get_random_samples, plot_audio_segments, plot_dendrogram
from clustering_utils import statistical_report,get_representative_calls_by_percentile, create_statistical_report_with_radar_plots, plot_and_save_audio_segments, plot_and_save_extreme_calls



features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_without_jtfs'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'

audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset'

# Path to save the results
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_hierarchical_clustering_without_jtfs\\_3_clusters_\\percentile'
# Create the results directory if it doesn't exist
if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)


distance_model = pd.DataFrame(columns=['distance_to_closest_cluster', 'children'])
# Get a list of all CSV files in the directory
list_files = glob.glob(os.path.join(features_path, '*.csv'))

all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)

# Drop NaN values
all_data = all_data.dropna()

# scale data with StandardScaler on used features only
scaler = StandardScaler()
features = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec','recording', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)


n_clusters = 3

agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', compute_distances=True)

cluster_membership = agg.fit_predict(features_scaled)

# Assign cluster memberships
all_data['cluster_membership'] = cluster_membership

all_data.to_csv(os.path.join(clusterings_results_path, f'hierarchical_clustering_{n_clusters}_membership.csv'), index=False)

linkage_matrix = linkage(features_scaled, method='ward')
# Compute cluster centers
cluster_centers = np.array([features_scaled[all_data['cluster_membership'] == i].mean(axis=0) for i in range(n_clusters)])

# Calculate distances of all points to their cluster centers
distances_to_centers = cdist(features_scaled, cluster_centers, 'euclidean')
# Save the distances to the DataFrame
all_data['distance_to_center'] = [distances_to_centers[i, cluster] for i, cluster in enumerate(cluster_membership)]


# Save all_data with cluster membership and distances
all_data.to_csv(os.path.join(clusterings_results_path, f'hierarchical_clustering_{n_clusters}_distance_membership.csv'), index=False)

# Define the percentiles to analyze
percentiles = [5, 10, 15, 20,25, 30,35, 40,45,  50,55, 60,65, 70,75, 80,85, 90, 95, 100]


                # Select the columns of interest for means
features = [
    'Duration_call','F0 Mean', 'F0 Std', 'F0 Skewness', 'F0 Kurtosis', 
    'F0 Bandwidth', 'F0 1st Order Diff', 'F0 Slope', 'F0 Mag Mean', 
    'F1 Mag Mean', 'F2 Mag Mean', 'F1-F0 Ratio', 'F2-F0 Ratio', 
    'Spectral Centroid Mean', 'Spectral Centroid Std', 'RMS Mean', 
    'RMS Std', 'Slope', 'Attack_magnitude', 'Attack_time']


# assign the percentiles to the dataframe
all_data['percentile'] = pd.qcut(all_data['distance_to_center'], len(percentiles), labels=percentiles)

# export and save the dataframe with the percentiles
all_data.to_csv(os.path.join(clusterings_results_path, f'hierarchical_clustering_{n_clusters}_percentiles.csv'), index=False)

clustered_groups = all_data.groupby(['cluster_membership', 'percentile'])

# # Loop over each cluster
# for cluster in range(n_clusters):
#     cluster_data = all_data[all_data['cluster_membership'] == cluster]
    
#     # Group the data by percentiles within this cluster
#     percentile_groups = cluster_data.groupby('percentile')

#     # Loop over each feature
#     for feature in features:
#         mean_values = []
#         std_values = []
#         labels = []
#         avg_distances = []
        
#         # Loop over each percentile and get mean and std for the current feature
#         for percentile, group in percentile_groups:
#             mean_values.append(group[feature].mean())
#             std_values.append(group[feature].std())
#             avg_distances.append(group['distance_to_center'].mean())
#             labels.append(f'{percentile} ({group["distance_to_center"].mean():.2f})')

#         # Plot the mean and std deviation for the current feature
#         plt.figure(figsize=(10, 6))
#         ax = sns.boxplot(x=labels, y=mean_values, palette='colorblind')
        
#         # Add error bars for standard deviation
#         ax.errorbar(x=labels, y=mean_values, yerr=std_values, fmt='none', c='black', capsize=5)
        
#         plt.title(f'Cluster {cluster} - Feature: {feature} - Across Percentiles')
#         plt.xlabel('Percentile (Avg. Distance)')
#         plt.ylabel('Mean Value')
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.legend(['Mean Value', 'Standard Deviation'], loc='upper right')
        
#         # Save and show the plot
#         plt.savefig(os.path.join(clusterings_results_path, f'cluster_{cluster}_feature_{feature}_percentiles.png'))
#         # plt.show()
#         plt.close()

# print("Plots with mean and standard deviation for each feature across different percentiles saved.")



# Loop over each cluster
for cluster in range(n_clusters):
    cluster_data = all_data[all_data['cluster_membership'] == cluster]
    percentile_groups = cluster_data.groupby('percentile')

    # Loop over each feature
    for feature in features:
        # Initialize plot
        plt.figure(figsize=(12, 8))

        # Create boxplot
        sns.boxplot(x='percentile', y=feature, data=cluster_data, color='lightblue')  

        # Add jittered points on top of boxplot
        sns.stripplot(x='percentile', y=feature, data=cluster_data, color='black', size=3, jitter=True, alpha=0.7)

        # Compute and annotate average distance for each percentile
        avg_distances = cluster_data.groupby('percentile')['distance_to_center'].mean().round(2)
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


# Selection of representative calls for each cluster
for cluster in range(n_clusters):
    cluster_mask = all_data['cluster_membership'] == cluster
    cluster_data = all_data[cluster_mask]
    
    # Sort data by distance to center (from closest to farthest)
    cluster_data = cluster_data.sort_values('distance_to_center', ascending=True)
    
    # Extract calls at specified percentiles
    calls_at_percentiles = get_representative_calls_by_percentile(cluster_data, percentiles)

    
    for percentile, calls in zip(percentiles, calls_at_percentiles):
        # Save path for representative calls
        save_path = os.path.join(clusterings_results_path, f'cluster_{cluster}_percentile_{percentile}')
        os.makedirs(save_path, exist_ok=True)
        
        # Save representative calls
        calls.to_csv(os.path.join(save_path, f'representative_calls_cluster_{cluster}_percentile_{percentile}.csv'), index=False)
        
        # Plot and save audio segments
        plot_and_save_audio_segments(calls, audio_path, save_path, f'cluster_{cluster}_percentile_{percentile}')
        
        print(f"\nRepresentative calls for cluster {cluster} at percentile {percentile}:")
        print(calls[['recording', 'call_id', 'distance_to_center', 'cluster_membership']])

print("Selection of calls at specified percentiles completed.")














# # save three csv files with the cluster membership
# for cluster in range(n_clusters):
#     cluster_mask = all_data['cluster_membership'] == cluster
#     # sort cluster data by distance to center from the closest to the farthest
#     cluster_data = all_data[cluster_mask].sort_values('distance_to_center', ascending=True)
#     cluster_data.to_csv(os.path.join(clusterings_results_path, f'hierarchical_clustering_{n_clusters}_cluster_{cluster}.csv'), index=False)


# # Identify the most and least representative points
# closest_points = []
# farthest_points = []

# for i in range(n_clusters):
#     cluster_mask = all_data['cluster_membership'] == i
#     cluster_data = all_data[cluster_mask]
#     cluster_distances = distances_to_centers[cluster_mask, i]
    
#     # Find indices of the closest and farthest points
#     closest_indices = np.argsort(cluster_distances)[:50]
#     farthest_indices = np.argsort(cluster_distances)[-50:]
    
#     # Add closest points
#     for idx in closest_indices:
#         point_data = cluster_data.iloc[idx].to_dict()
#         point_data['distance'] = cluster_distances[idx]
#         point_data['point_type'] = 'closest'
#         closest_points.append(point_data)
    
#     # Add farthest points
#     for idx in farthest_indices:
#         point_data = cluster_data.iloc[idx].to_dict()
#         point_data['distance'] = cluster_distances[idx]
#         point_data['point_type'] = 'farthest'
#         farthest_points.append(point_data)

# # Combine and save results
# representative_calls_df = pd.DataFrame(closest_points + farthest_points)
# representative_calls_df.to_csv(os.path.join(clusterings_results_path, 'hierarchical_representative_calls.csv'), index=False)





# count calls for each cluster
# calls_per_cluster = all_data['cluster_membership'].value_counts().sort_index()


# def get_representative_calls(cluster_data, start_rank, n_calls=25):
#     return cluster_data.iloc[start_rank:start_rank+n_calls]

# Selection of representative calls for each cluster
# for cluster in range(n_clusters):
#     cluster_mask = all_data['cluster_membership'] == cluster
#     cluster_data = all_data[cluster_mask]
#     cluster_data = cluster_data.sort_values('distance_to_center', ascending=False)
    
#     # Save sorted cluster data
#     cluster_data.to_csv(os.path.join(clusterings_results_path, f'hierarchical_clustering_{n_clusters}_cluster_{cluster}.csv'), index=False)
    
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
#         print(representative_calls[['recording', 'call_id', 'distance_to_center']])
#         print("\nPlease analyse these calls.")
#         response = input("Have you found the threshold in this batch? (yes/no): ")
    
#         if response.lower() == 'yes':
#             threshold_rank = int(input("Enter the rank number where you found the threshold: "))
#             threshold_distance = cluster_data.iloc[threshold_rank-1]['distance_to_center']
#             print(f"Threshold found for cluster {cluster} at rank {threshold_rank}, distance {threshold_distance:.4f}")
#             threshold_found = True
#         else:
#             start_rank += len(representative_calls)
    
#     # After finding threshold, save it
#     if threshold_found:
#         threshold_path = os.path.join(clusterings_results_path, f'cluster_{cluster}_threshold.txt')
#         with open(threshold_path, 'w') as f:
#             f.write(f"Threshold for cluster {cluster}: rank {threshold_rank}\n")
#             f.write(f"This corresponds to a distance_to_center of {threshold_distance:.4f}\n")

# print("Hierarchical clustering: Thresholds for each cluster determined.")

# # Count calls for each cluster
# calls_per_cluster = all_data['cluster_membership'].value_counts().sort_index()
# print("Calls per cluster:")
# print(calls_per_cluster)














# calculate the distances for each point 
# The linkage matrix is a matrix of shape (n-1, 4) where n is the number of original observations in the data.
# It is used to store the hierarchical clustering result.
# Each row of the z\ Linkage matrix) represents a merger in the hierarchical clustering process and contains the following information:
# 1. Z[i, 0]: The index of the first cluster that is merged.
# 2. Z[i, 1]: The index of the second cluster being merged.
# 3. Z[i, 2]: The distance between the two clusters being merged.
# 4. Z[i, 3]: The number of original observations in the newly formed cluster
# THIS BELOW IS THE CODE TO CALCULATE THE DISTANCES FOR EACH POINT TO THE CLOSEST CLUSTER THROUGH THE LINKAGE MATRIX

# # Calculate distances for all points
# def compute_distances(linkage_matrix, n_samples):
#     distances = np.zeros(n_samples)
#     for i in range(n_samples - 1):
#         cluster1 = int(linkage_matrix[i, 0])
#         cluster2 = int(linkage_matrix[i, 1])
#         distance = linkage_matrix[i, 2]
#         if cluster1 < n_samples:
#             distances[cluster1] = distance
#         if cluster2 < n_samples:
#             distances[cluster2] = distance
#     return distances

# distances = compute_distances(linkage_matrix, len(all_data))
# all_data['distance_to_closest_cluster'] = distances

# all_data.to_csv(os.path.join(clusterings_results_path, f'hierarchical_clustering_{n_clusters}_membership.csv'), index=False)







# Perform DENDROGRAM for obtaining the standard embeddings for the clustering techniques
# # Plot the dendrogram and get the cluster memberships
# membership = plot_dendrogram(agg, num_clusters=n_clusters)
# if membership is not None:
#     print(membership)

# # Get 5 random samples for each cluster
# random_samples = get_random_samples(all_data, 'cluster_membership', num_samples=5)
# # Plot the audio segments and save audio files
# plot_and_save_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')

# # Plot the audio segments
# # plot_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')

# # Get the statistical report
# stats = statistical_report(all_data, cluster_membership,n_clusters, metadata, clusterings_results_path)
# print(stats)
# radar= statistical_report_df = create_statistical_report_with_radar_plots(all_data, cluster_membership, n_clusters, metadata, clusterings_results_path)





