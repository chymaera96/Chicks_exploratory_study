import os
import glob
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
from kneed import KneeLocator
from clustering_utils import find_elbow_point
import itertools
# Path to the directory containing the CSV files
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_'

# Path to the metadata CSV file
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'

#save the results
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering\\_gmm_clustering_'
if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)


# Get a list of all CSV files in the directory
list_files = glob.glob(os.path.join(features_path, '*.csv'))

# Path to save the results
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering\\_gmm_clustering_'
# Check if the directory exists, if not, create it
if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)

# Read all CSV files and concatenate them into a single DataFrame
all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)

# Drop NaN values
all_data = all_data.dropna()

# Scale data with StandardScaler on used features only
scaler = StandardScaler()
features = all_data.drop(['recording','Call Number', 'onsets_sec', 'offsets_sec'], axis=1)
features_scaled = scaler.fit_transform(features)


# 1) Gaussian Mixture Model (GMM) clustering
n_max_components = (len(features.columns) - 1) * 2
components= range(2,11)

covarience_type =  'full'
n_init= ( 1, 5, 50)
max_iter=1000
toll = 1e-3
random_state=42
init_params = 'kmeans'


gmm_cluster_evaluation_per_number_clusters = {n_components: {'silhouette_score': 0, 'calinski_harabasz_score': 0, 'wcss':999, 'bic': 0, 'aic': 0} for n_components in itertools.product(components, n_init)}
for n_comp, n_init  in itertools.product(components, n_init):
    gmm = GaussianMixture(n_components=n_comp, covariance_type=covarience_type, n_init=n_init, max_iter=max_iter, tol=toll, random_state=random_state, init_params=init_params) 
    cluster_membership = gmm.fit_predict(features_scaled)

    number_of_clusters = len(set(cluster_membership)) - (1 if -1 in cluster_membership else 0)  
    # save the number of clusters
    gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['number_of_cluster'] = number_of_clusters
    # save the parameters
    gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['n_components'] = n_comp, n_init


    # compute the centroid of each cluster
    centroids= []
    for cluster in range(n_comp):
        centroids.append(np.mean(features_scaled[cluster_membership == cluster], axis=0))
    cluster_center = np.array(centroids)

    gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['silhouette_score'] = silhouette_score(features_scaled, cluster_membership)
    gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['calinski_harabasz_score'] = calinski_harabasz_score(features_scaled, cluster_membership)

    gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['wcss'] = np.sum((np.linalg.norm(features_scaled - cluster_center[cluster_membership], axis=1) ** 2))
    gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['bic'] = gmm.bic(features_scaled)
    gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['aic'] = gmm.aic(features_scaled)

    print('Number of clusters:', number_of_clusters) 
    print('Silhouette score:', gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['silhouette_score'])
    print('Calinski Harabasz score:', gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['calinski_harabasz_score'])
    print('BIC:', gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['bic'])
    print('AIC:', gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['aic'])
    print()

# Save results in the Results_Clustering folder
gmm_cluster_evaluation_per_number_clusters_df = pd.DataFrame(gmm_cluster_evaluation_per_number_clusters).T
gmm_cluster_evaluation_per_number_clusters_df.to_csv(os.path.join(clusterings_results_path, 'gmm_cluster_evaluation_per_number_clusters.csv'), index=False)

# convert and export the results to latex
gmm_cluster_evaluation_per_number_clusters_df.to_latex(os.path.join(clusterings_results_path, 'gmm_cluster_evaluation_per_number_clusters.tex'))

# Find the vector of silhouette scores, calinski harabasz scores, wcss, bic, and aic across the number of clusters
wc_ss_vector_across_n_clusters = [gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['wcss'] for n_comp, n_init in gmm_cluster_evaluation_per_number_clusters.keys()]
bic_vector_across_n_clusters = [gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['bic'] for n_comp, n_init in gmm_cluster_evaluation_per_number_clusters.keys()]
aic_vector_across_n_clusters = [gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['aic'] for n_comp, n_init in gmm_cluster_evaluation_per_number_clusters.keys()]

# Find the vector of silhouette scores, calinski harabasz scores, wcss, bic, and aic across the number of clusters
wc_ss_vector_across_n_clusters = [gmm_cluster_evaluation_per_number_clusters[number_of_clusters]['wcss'] for number_of_clusters in gmm_cluster_evaluation_per_number_clusters.keys()]
bic_vector_across_n_clusters = [gmm_cluster_evaluation_per_number_clusters[n_clusters]['bic'] for n_clusters in gmm_cluster_evaluation_per_number_clusters.keys()]
aic_vector_across_n_clusters = [gmm_cluster_evaluation_per_number_clusters[n_clusters]['aic'] for n_clusters in gmm_cluster_evaluation_per_number_clusters.keys()]

# fIND THE ELBOW POINTS
wcss_elbow = KneeLocator(range(2, n_max_components), wc_ss_vector_across_n_clusters, curve='convex', direction='decreasing', interp_method='polynomial', online=False)
bic_elbow = KneeLocator(range(2, n_max_components), bic_vector_across_n_clusters, curve='convex', direction='decreasing', interp_method='polynomial', online=False)
aic_elbow = KneeLocator(range(2, n_max_components), aic_vector_across_n_clusters,curve='convex', direction='decreasing', interp_method='polynomial', online=False)

best_n_clusters_by_wcss_elbow = wcss_elbow.elbow
best_n_clusters_by_bic_elbow = bic_elbow.elbow
best_n_clusters_by_aic_elbow = aic_elbow.elbow


# best_n_clusters_by_wcss_elbow = find_elbow_point(wc_ss_vector_across_n_clusters)
# best_n_clusters_by_bic_elbow = find_elbow_point(bic_vector_across_n_clusters)
# best_n_clusters_by_aic_elbow = find_elbow_point(aic_vector_across_n_clusters)


plt.figure(figsize=(25, 6))

# Plot Silhouette score
plt.subplot(1, 2, 1)
# plot number of clusters vs silhouette score
# save the list of number of clusters and silhouette scores
number_of_clusters = [gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['number_of_cluster'] for n_comp, n_init in gmm_cluster_evaluation_per_number_clusters.keys()]
silhouette_scores = [gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['silhouette_score'] for n_comp, n_init in gmm_cluster_evaluation_per_number_clusters.keys()]

plt.plot(number_of_clusters, silhouette_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score per number of clusters')

# Plot Calinski Harabasz score
plt.subplot(1, 2, 2)
calinski_harabasz_scores = [gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['calinski_harabasz_score'] for n_comp, n_init in gmm_cluster_evaluation_per_number_clusters.keys()]
plt.plot(number_of_clusters, calinski_harabasz_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Calinski Harabasz score')
plt.title('Calinski Harabasz score per number of clusters')

plt.tight_layout()
plt.show()
plt.savefig('gmm_clustering_evaluation1.png')

# Create a new figure for BIC, AIC, and WCSS
plt.figure(figsize=(25, 6))

# plot the BIC, AIC, and WCSS per number of clusters
plt.subplot(1, 3, 1)
wcss_values = [gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['wcss'] for n_comp, n_init in gmm_cluster_evaluation_per_number_clusters.keys()]
plt.plot(number_of_clusters, wcss_values, 'bx-')
plt.axvline(x=best_n_clusters_by_wcss_elbow, color='r', linestyle='--', label=f'Elbow point: {best_n_clusters_by_wcss_elbow}')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('WCSS per number of clusters')
plt.legend()

plt.subplot(1, 3, 2)
bic_values = [gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['bic'] for n_comp, n_init in gmm_cluster_evaluation_per_number_clusters.keys()]
plt.plot(number_of_clusters, bic_values, 'bx-')
plt.axvline(x=best_n_clusters_by_bic_elbow, color='r', linestyle='--', label=f'Elbow point: {best_n_clusters_by_bic_elbow}')
plt.xlabel('Number of clusters')
plt.ylabel('BIC')
plt.title('BIC per number of clusters')
plt.legend()

# Plot AIC
plt.subplot(1, 3, 3)
aic_values = [gmm_cluster_evaluation_per_number_clusters[n_comp, n_init]['aic'] for n_comp, n_init in gmm_cluster_evaluation_per_number_clusters.keys()]
plt.plot(number_of_clusters, aic_values, 'bx-')
# signal the optimal number of clusters with a red circle
plt.axvline(x=best_n_clusters_by_aic_elbow, color='r', linestyle='--', label= f'Elbow point: {best_n_clusters_by_aic_elbow}')
plt.xlabel('Number of clusters')
plt.ylabel('AIC')
plt.title('AIC per number of clusters')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('gmm_clustering_evaluation2.png')
print('Gaussian Mixture Model (GMM) clustering done')

