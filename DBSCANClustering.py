import os
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
from kneed import KneeLocator
from gap_statistic import OptimalK
import itertools

# Path to the directory containing the CSV files
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_dbscan_clustering_'

#save the results
if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)

# Get a list of all CSV files in the directory
list_files = glob.glob(os.path.join(features_path, '*.csv'))

# Read all CSV files and concatenate them into a single DataFrame
all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)

# Drop NaN values
all_data = all_data.dropna()

# Scale data with StandardScaler on used features only
scaler = StandardScaler()
features = all_data.drop(['recording', 'Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)

# Parameters ranges
eps_range = np.linspace(4.0, 9.0, 30)
min_samples_range =[ 2, 3, 4, 5 ,6, 7, 8, 9, 10, 20, 30 , 40, 50, 60, 70, 80, 90, 100]

dbscan_cluster_evaluation_per_params = {
    params: {
        'eps': 0, 'min_samples': 0, 'number_of_cluster': 0,
        'silhouette_score': 0, 'calinski_harabasz_score': 0, 'wcss': 0
    } for params in itertools.product(eps_range, min_samples_range)
}



# Iterate over the combinations of eps and min_samples
for e, pts in itertools.product(eps_range, min_samples_range):
    # Perform DBSCAN on the scaled data
    dbscan = DBSCAN(eps=e, min_samples=pts)
    cluster_membership = dbscan.fit_predict(features_scaled)


    # Save the parameters eps and min_samples
    params = (e, pts)
    # save the eps and min_samples
    dbscan_cluster_evaluation_per_params[params]['eps'] = e
    dbscan_cluster_evaluation_per_params[params]['min_samples'] = pts
    dbscan_cluster_evaluation_per_params[params]['cluster_membership'] = cluster_membership


          
    noise_points = []
    # Compute the number of clusters
    number_of_clusters = len(set(cluster_membership)) - (1 if -1 in cluster_membership else 0)
    dbscan_cluster_evaluation_per_params[params]['number_of_cluster'] = number_of_clusters
    dbscan_cluster_evaluation_per_params[params]['noise_points'] = noise_points

    features_scaled_no_noise = features_scaled[cluster_membership != -1]
    cluster_membership_no_noise = cluster_membership[cluster_membership != -1]

    if number_of_clusters > 1:
        # Compute the centroid of each cluster
        centroids = []
        for cluster in range(number_of_clusters):
            centroids.append(np.mean(features_scaled_no_noise[cluster_membership_no_noise == cluster], axis=0))
        cluster_center = np.array(centroids)

        dbscan_cluster_evaluation_per_params[params]['silhouette_score'] = silhouette_score(features_scaled_no_noise, cluster_membership_no_noise)
        dbscan_cluster_evaluation_per_params[params]['calinski_harabasz_score'] = calinski_harabasz_score(features_scaled_no_noise, cluster_membership_no_noise)
        dbscan_cluster_evaluation_per_params[params]['wcss'] = np.sum(np.linalg.norm(features_scaled_no_noise - cluster_center[cluster_membership_no_noise], axis=1) ** 2)
    else:
        dbscan_cluster_evaluation_per_params[params]['silhouette_score'] = 0
        dbscan_cluster_evaluation_per_params[params]['calinski_harabasz_score'] = 0
        dbscan_cluster_evaluation_per_params[params]['wcss'] = 0

    print(f'eps: {e}, min_samples: {pts}, number of clusters: {number_of_clusters}, '
          f'silhouette_score: {dbscan_cluster_evaluation_per_params[params]["silhouette_score"]}, '
          f'calinski_harabasz_score: {dbscan_cluster_evaluation_per_params[params]["calinski_harabasz_score"]}, '
          f'wcss: {dbscan_cluster_evaluation_per_params[params]["wcss"]}')
# Remove the combinations of eps and min_samples that have only one cluster
dbscan_cluster_evaluation_per_params = {
    k: v for k, v in dbscan_cluster_evaluation_per_params.items() if v['number_of_cluster'] > 1
}


dbscan_cluster_evaluation_per_params_df = pd.DataFrame(dbscan_cluster_evaluation_per_params).T
dbscan_cluster_evaluation_per_params_df = dbscan_cluster_evaluation_per_params_df.sort_values(by='number_of_cluster', ascending=True)
dbscan_cluster_evaluation_per_params_df.to_csv(os.path.join(clusterings_results_path, 'dbscan_cluster_evaluation_per_params.csv'), index=False)
dbscan_cluster_evaluation_per_params_df.to_latex(os.path.join(clusterings_results_path, 'dbscan_cluster_evaluation_per_params.tex'))

# Find the optimal number of clusters using the OptimalK class
optimal_k = OptimalK(parallel_backend='joblib')
n_clusters_optimal_k = optimal_k(features_scaled, cluster_array=np.arange(2, 11))
print('Optimal number of clusters for k:', n_clusters_optimal_k)

# Ensure the values are converted to float64
wcss_values = dbscan_cluster_evaluation_per_params_df['wcss'].astype(float).values
# Create the KneeLocator object
wcss_knee_finder = KneeLocator(range(2, len(wcss_values) + 2), wcss_values, curve='convex', direction='decreasing', interp_method='polynomial', online=False)
wcss_elbow = wcss_knee_finder.elbow

number_of_clusters = dbscan_cluster_evaluation_per_params_df['number_of_cluster'].values
silhouette_scores = dbscan_cluster_evaluation_per_params_df['silhouette_score'].values

max_silhouette_score = dbscan_cluster_evaluation_per_params_df['silhouette_score'].max()
max_silhouette_score_params = dbscan_cluster_evaluation_per_params_df[
    dbscan_cluster_evaluation_per_params_df['silhouette_score'] == max_silhouette_score]

calinski_harabasz_scores = dbscan_cluster_evaluation_per_params_df['calinski_harabasz_score'].values
max_calinski_harabasz_score = dbscan_cluster_evaluation_per_params_df['calinski_harabasz_score'].max()
max_calinski_harabasz_score_params = dbscan_cluster_evaluation_per_params_df[
    dbscan_cluster_evaluation_per_params_df['calinski_harabasz_score'] == max_calinski_harabasz_score]

eps_vector = dbscan_cluster_evaluation_per_params_df['eps'].values
min_samples_vector = dbscan_cluster_evaluation_per_params_df['min_samples'].values
cluster_vector = dbscan_cluster_evaluation_per_params_df['number_of_cluster'].values



# # Trova il miglior valore di Silhouette score per ogni numero di cluster nel range specificato

# cluster_founded = dbscan_cluster_evaluation_per_params_df['number_of_cluster'].unique()
# best_silhouette_per_cluster = []
# for num_clusters in cluster_founded:  # selct the max silhouette score for each number of cluster
#     silhouette_scores_for_cluster = dbscan_cluster_evaluation_per_params_df[dbscan_cluster_evaluation_per_params_df['number_of_cluster'] == num_clusters]['silhouette_score'].values
#     best_silhouette_per_cluster.append(silhouette_scores_for_cluster.max())



# best_calinski_per_cluster = []
# for num_clusters in cluster_founded:
#     calinski_scores_for_cluster = dbscan_cluster_evaluation_per_params_df[dbscan_cluster_evaluation_per_params_df['number_of_cluster'] == num_clusters]['calinski_harabasz_score'].values
#     best_calinski_per_cluster.append(calinski_scores_for_cluster.max())


# best_wcss_per_cluster = []
# for num_clusters in cluster_founded:
#     wcss_for_cluster = dbscan_cluster_evaluation_per_params_df[dbscan_cluster_evaluation_per_params_df['number_of_cluster'] == num_clusters]['wcss'].values
#     best_wcss_per_cluster.append(wcss_for_cluster.max())




# print(cluster_founded)

# plt.figure(figsize=(30, 5))

# # Plot del miglior Silhouette score per numero di cluster
# plt.subplot(1, 3, 1)
# plt.plot(range(2, len(cluster_founded) + 2), best_silhouette_per_cluster, 'bx-')
# plt.xlabel('Number of clusters')
# plt.ylabel('Best Silhouette score')
# plt.title('Best Silhouette score per number of clusters')
# plt.grid()

# # Plot del miglior Calinski Harabasz score per numero di cluster
# plt.subplot(1, 3, 2)
# plt.plot(range(2, len(cluster_founded) + 2), best_calinski_per_cluster, 'bx-')
# plt.xlabel('Number of clusters')
# plt.ylabel('Best Calinski Harabasz score')
# plt.title('Best Calinski Harabasz score per number of clusters')
# plt.grid()

# # Plot del miglior WCSS per numero di cluster
# plt.subplot(1, 3, 3)
# plt.plot(range(2, len(cluster_founded) + 2), best_wcss_per_cluster, 'bx-')
# plt.axvline(x=wcss_elbow, color='r', linestyle='--', label=f'Elbow point: {wcss_elbow}')
# plt.xlabel('Number of clusters')
# plt.ylabel('Best WCSS')
# plt.title('Best WCSS per number of clusters')
# plt.legend()
# plt.grid()

# # Adjust layout and save the figure
# plt.tight_layout()
# plt.savefig('C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_dbscan_clustering_\\dbscan_cluster_evaluation_per_params.png')
# plt.show()




# Trova il miglior valore di Silhouette score per ogni numero di cluster nel range specificato
cluster_founded = sorted(dbscan_cluster_evaluation_per_params_df['number_of_cluster'].unique())
best_silhouette_per_cluster = []
best_calinski_per_cluster = []
best_wcss_per_cluster = []

for num_clusters in cluster_founded:
    silhouette_scores_for_cluster = dbscan_cluster_evaluation_per_params_df[
        dbscan_cluster_evaluation_per_params_df['number_of_cluster'] == num_clusters]['silhouette_score'].values
    best_silhouette_per_cluster.append(silhouette_scores_for_cluster.max())

    calinski_scores_for_cluster = dbscan_cluster_evaluation_per_params_df[
        dbscan_cluster_evaluation_per_params_df['number_of_cluster'] == num_clusters]['calinski_harabasz_score'].values
    best_calinski_per_cluster.append(calinski_scores_for_cluster.max())

    wcss_for_cluster = dbscan_cluster_evaluation_per_params_df[
        dbscan_cluster_evaluation_per_params_df['number_of_cluster'] == num_clusters]['wcss'].values
    best_wcss_per_cluster.append(wcss_for_cluster.max())

# Plot
plt.figure(figsize=(30, 5))

# Plot del miglior Silhouette score per numero di cluster
plt.subplot(1, 3, 1)
plt.plot(cluster_founded, best_silhouette_per_cluster, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Best Silhouette score')
plt.title('Best Silhouette score per number of clusters')
plt.grid()

# Plot del miglior Calinski Harabasz score per numero di cluster
plt.subplot(1, 3, 2)
plt.plot(cluster_founded, best_calinski_per_cluster, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Best Calinski Harabasz score')
plt.title('Best Calinski Harabasz score per number of clusters')
plt.grid()

# Plot del miglior WCSS per numero di cluster
plt.subplot(1, 3, 3)
plt.plot(cluster_founded, best_wcss_per_cluster, 'bx-')
wcss_elbow = cluster_founded[best_wcss_per_cluster.index(min(best_wcss_per_cluster))] # esempio di punto gomito
plt.axvline(x=wcss_elbow, color='r', linestyle='--', label=f'Elbow point: {wcss_elbow}')
plt.xlabel('Number of clusters')
plt.ylabel('Best WCSS')
plt.title('Best WCSS per number of clusters')
plt.legend()
plt.grid()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_dbscan_clustering_\\dbscan_cluster_evaluation_per_params.png')
plt.show()
