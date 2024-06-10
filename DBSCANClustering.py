import os
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
from kneed import KneeLocator
import itertools
from clustering_utils import find_elbow_point_wcss

# Path to the directory containing the CSV files
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering\\_dbscan_clustering_'

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
features = all_data.drop(['recording', 'Call Number', 'onsets_sec', 'offsets_sec'], axis=1)
features_scaled = scaler.fit_transform(features)

# Parameters ranges
eps_range = np.linspace(4.0, 7.0, 20)
min_samples_range = range(2, 5, 10, 20, 30 , 40, 50, 60, 70, 80, 90, 100)

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
dbscan_cluster_evaluation_per_params_df = dbscan_cluster_evaluation_per_params_df.sort_values(by='number_of_cluster')
dbscan_cluster_evaluation_per_params_df.to_csv(os.path.join(clusterings_results_path, 'dbscan_cluster_evaluation_per_params.csv'), index=False)
dbscan_cluster_evaluation_per_params_df.to_latex(os.path.join(clusterings_results_path, 'dbscan_cluster_evaluation_per_params.tex'))

wcss_values = dbscan_cluster_evaluation_per_params_df['wcss'].values

wcss_knee_finder = KneeLocator(range(1, len(wcss_values) + 1), wcss_values, curve='convex', direction='decreasing', interp_method='polynomial', online=False)
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

# # Plotting
# plt.figure(figsize=(25, 12))

# # Plot Silhouette score
# plt.subplot(1, 2, 1)
# plt.plot(cluster_vector, silhouette_scores)
# plt.xlabel('Number of combinations of eps and min_samples')
# plt.ylabel('Silhouette score')
# plt.title('Silhouette score per combination of eps and min_samples')
# plt.grid()
# plt.axvline(x=max_silhouette_score, color='r', linestyle='--', label='Best Silhouette score')
# plt.legend()

# # Plot Calinski Harabasz score
# plt.subplot(1, 2, 2)
# plt.plot(cluster_vector, calinski_harabasz_scores)
# plt.xlabel('Number of combinations of eps and min_samples')
# plt.ylabel('Calinski Harabasz score')
# plt.title('Calinski Harabasz score per combination of eps and min_samples')
# plt.grid()
# plt.axvline(x=max_calinski_harabasz_score, color='r', linestyle='--', label='Best CH score')
# plt.legend()

# plt.savefig(os.path.join(clusterings_results_path, 'dbscan_cluster_evaluation_per_params.png'))

# # Plot the WCSS per number of clusters
# plt.figure(figsize=(20, 4))
# plt.plot(cluster_vector, wcss_values)
# plt.xlabel('Number of combinations of eps and min_samples')
# plt.ylabel('WCSS')
# plt.title('WCSS per number of combinations of eps and min_samples')
# plt.grid()
# plt.axvline(x=wcss_elbow, color='r', linestyle='--', label=f'Elbow point: {wcss_elbow}')
# plt.legend()
# plt.savefig(os.path.join(clusterings_results_path, 'dbscan_wcss.png'))
# plt.show()



# Save results
results = pd.DataFrame({
    'eps': [eps_vector],
    'min_samples': [min_samples_vector],
    'number_of_clusters': [number_of_clusters],
    'silhouette_score': [silhouette_scores],
    'calinski_harabasz_score': [calinski_harabasz_scores],
    'wcss': [wcss_values]
})
results.to_csv(os.path.join(clusterings_results_path, 'dbscan_cluster_final_evaluation.csv'), index=False)

# Plotting
plt.figure(figsize=(12, 6))

# Plot Silhouette score
plt.subplot(3, 1, 1)
plt.plot(eps_vector, silhouette_scores)
plt.xlabel('eps')
plt.ylabel('Silhouette score')
plt.title('Silhouette score for eps')
plt.grid()

# Plot Calinski Harabasz score
plt.subplot(3, 1, 2)
plt.plot([eps_vector], [calinski_harabasz_scores], marker='o')
plt.xlabel('eps')
plt.ylabel('Calinski Harabasz score')
plt.title('Calinski Harabasz score for eps')
plt.grid()

# Plot WCSS
plt.subplot(3, 1, 3)
plt.plot(eps_vector, wcss_values)
plt.xlabel('eps')
plt.ylabel('WCSS')
plt.axline((eps_vector[wcss_elbow], wcss_values[wcss_elbow]), (eps_vector[wcss_elbow], 0), color='r', linestyle='--', label=f'Elbow point: {eps_vector[wcss_elbow]}')
plt.title('WCSS for eps')
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(clusterings_results_path, 'dbscan_cluster_evaluation.png'))
plt.show()

print(f'Best parameters: eps = {max_silhouette_score_params["eps"].values[0]}, min_samples = {max_silhouette_score_params["min_samples"].values[0]}')
print(f'Number of clusters: {number_of_clusters}')
print(f'Silhouette score: {max_silhouette_score}')
print(f'Calinski Harabasz score: {max_calinski_harabasz_score}')
print(f'WCSS: {wcss_values[wcss_elbow]}')