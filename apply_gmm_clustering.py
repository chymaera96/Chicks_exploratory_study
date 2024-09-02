
import os
import glob
import pandas as pd
import numpy as np
import umap.umap_ as umap
# import umap  #install umap-learn
import matplotlib.pyplot as plt
from kneed import KneeLocator
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score
from sklearn.metrics import calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture

from sklearn.cluster import AgglomerativeClustering
from clustering_utils import get_random_samples, plot_audio_segments
from clustering_utils import get_random_samples, plot_audio_segments, statistical_report, create_statistical_report_with_radar_plots, plot_and_save_audio_segments, plot_and_save_extreme_calls


# features_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/_results_high_quality_dataset_'
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_without_jtfs'

# metadata_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/High_quality_dataset/high_quality_dataset_metadata.csv'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'

audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset'

# Path to save the results
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_gmm_clustering_without_jtfs_'
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
features = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)


n_components = 3

gmm = GaussianMixture(n_components=n_components, random_state=42)
cluster_membership = gmm.fit_predict(features_scaled)

all_data['cluster_membership'] = cluster_membership
# save the results
all_data.to_csv(os.path.join(clusterings_results_path, f'gaussian_mixture_2_membership.csv'), index=False)  


# Perform UMAP for obtaining the standard embeddings for the clustering techniques
umap_reducer = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7)
standard_embedding = umap_reducer.fit_transform(features_scaled)

# Plot the UMAP embeddings with the cluster membership
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

custom_colors = ['dodgerblue','Green','LightCoral', 'Cyan', 'MediumSlateBlue', 'ForestGreen', 'DarkTurquoise', 'DarkSlateBlue','LightSalmon', 'MediumPurple', 'Magenta', 'Yellow', 'LightCoral', 'MediumSlateBlue', 'ForestGreen', 'DarkTurquoise', 'DarkSlateBlue']

for j in range(n_components):
    ax.scatter(standard_embedding[cluster_membership == j, 0], 
               standard_embedding[cluster_membership == j, 1], 
               standard_embedding[cluster_membership == j, 2], c=custom_colors[j], s=10, label=f'Cluster {j}', alpha=0.3)
ax.legend(loc='upper right', markerscale=2)

plt.title(f"UMAP projection of chicks' calls clustered with Gaussian Mixture Model (n={n_components})", fontsize=10)
plt.legend()
plt.show()

#extract segments from the audio files
random_samples = get_random_samples(all_data, 'cluster_membership', num_samples=5)
print('Random samples selected')

plot_and_save_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')

radar_results= create_statistical_report_with_radar_plots(all_data, cluster_membership, n_components, metadata, clusterings_results_path)
# # Plot the audio segments
# plot_audio_segments(random_samples, audio_path, clusterings_results_path, f'gaussian_mixture_{n_components}_membership')
print('Audio segments plotted')


# This part of the code is commented out because it is not necessary for the clustering techniques
# the aim below is to extract the probabilities of each sample to belong to each cluster
# and to use these probabilities to determine which samples are well-clustered and which are not

# # Extract probabilities from GMM
# feateures_scaled = gmm._validate_data(features_scaled, reset=False)

# weighted_probs = gmm._estimate_weighted_log_prob(features_scaled)

# unnormalized_probabilities = np.exp(weighted_probs)

# #Predict posterior probability of data under each Gaussian in the model.
# p_probs = gmm.predict_proba(features_scaled)

# # Compute the log probability under the model.
# log_probs = gmm.score(features_scaled)

# print('Log Probabilities:', log_probs)

# #Return the per-sample likelihood of the data under the model.
# likelihoods = gmm.score_samples(features_scaled)


# all_data['prob_0'] = p_probs[:, 0]
# all_data['prob_1'] = p_probs[:, 1]
# all_data['prob_2'] = p_probs[:, 2]


# all_data['not_norm_prob_0'] = unnormalized_probabilities[:, 0]
# all_data['not_norm_prob_1'] = unnormalized_probabilities[:, 1]


# all_data['likelihood'] = likelihoods



# # save and export to csv
# all_data.to_csv(os.path.join(clusterings_results_path, f'__gaussian_mixture_{n_components}_membership_with_probabilities.csv'), index=False)


# # (Assumendo che "bene assegnate" significhi alte probabilità di appartenenza al cluster predetto)
# threshold = 0.9  # Adatta questo valore in base ai tuoi requisiti

# # Chiamate ben assegnate (probabilità alta per il cluster predetto)
# well_clustered = all_data[(all_data['prob_0'] > threshold) | (all_data['prob_1'] > threshold)]

# # Chiamate non ben assegnate (probabilità bassa per il cluster predetto)
# not_well_clustered = all_data[(all_data['prob_0'] <= threshold) & (all_data['prob_1'] <= threshold)]

# # Salva le chiamate non ben assegnate come undefined
# not_well_clustered.to_csv(os.path.join(clusterings_results_path, 'undefined_calls.csv'), index=False)

# # Chiama la funzione plot_and_save_extreme_calls per entrambe le categorie
# plot_and_save_extreme_calls(well_clustered, audio_path, clusterings_results_path, 'well_clustered')
# plot_and_save_extreme_calls(not_well_clustered, audio_path, clusterings_results_path, 'undefined')

# print("Process completed and files saved.")


#_estimate_weighted_log_prob= Estimate the log-probabilities log P(X | Z).
#Compute the log-probabilities per each component for each sample.

# normalized_log_probs = weighted_probs - np.max(weighted_probs, axis=1)[:, np.newaxis]

# log_prob_norm, log_resp = gmm._estimate_log_prob_resp(features_scaled)
# responsibilities = np.exp(log_resp)


# stats = statistical_report(all_data, cluster_membership, n_components, metadata, clusterings_results_path)
# print(stats)
# radar= statistical_report_df = create_statistical_report_with_radar_plots(all_data, cluster_membership, n_components, metadata, clusterings_results_path)


# # Save the probabilities to belong to each cluster ( from 0 to 1) in the main dataframe
# all_data['prob_0'] = probabilities[:, 0]
# all_data['prob_1'] = probabilities[:, 1]


# # Save the log probabilities to belong to each cluster in the main dataframe
# all_data['log_prob_0'] = weighted_probs[:, 0]
# all_data['log_prob_1'] = weighted_probs[:, 1]

# # Funzione per verificare se una chiamata è ben assegnata
# def is_well_assigned(row):
#     if row['cluster_membership'] == 0:
#         return row['prob_0'] > row['prob_1']
#     elif row['cluster_membership'] == 1:
#         return row['prob_1'] > row['prob_0']
#     else:
#         return False


# # Applica i criteri per determinare se una chiamata è ben assegnata
# all_data['well_assigned'] = all_data.apply(is_well_assigned, axis=1)

# # Filtra le chiamate non ben assegnate
# not_well_assigned = all_data[~all_data['well_assigned']]

# # Salva le chiamate non ben assegnate in un file CSV
# # not_well_assigned.to_csv(os.path.join(clusterings_results_path, 'undefined_calls.csv'), index=False)


# # save the results with probabilities and export to csv
# all_data.to_csv(os.path.join(clusterings_results_path, f'gaussian_mixture_{n_components}_membership_with_probabilities.csv'), index=False)


# # Create a scatter plot
# plt.figure(figsize=(10, 8))
# # Plot points for cluster 0
# cluster_0 = all_data[all_data['cluster_membership'] == 0]
# plt.scatter(cluster_0['prob_1'], cluster_0['prob_0'], c='blue', alpha=0.5, s= 10, label='Cluster 0')
# # Plot points for cluster 1
# cluster_1 = all_data[all_data['cluster_membership'] == 1]
# plt.scatter(cluster_1['prob_1'], cluster_1['prob_0'], c='red', alpha=0.5, s= 10, label='Cluster 1')
# # Set labels and title
# plt.xlabel('Probability of belonging to Cluster 1')
# plt.ylabel('Probability of belonging to Cluster 0')
# plt.title('Cluster Membership Posterior Probabilities')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# # plt.xlim(0, 1)
# # plt.ylim(0, 1)
# plt.savefig(os.path.join(clusterings_results_path, 'cluster_membership_posterior_probabilities.png'))
# plt.show()


# # count the number of samples assigned to each cluster
# unique, counts = np.unique(cluster_membership, return_counts=True)
# cluster_counts = dict(zip(unique, counts))
# print("Cluster Counts:", cluster_counts)

# # Calculate the average probability of belonging to each cluster
# avg_probabilities = np.mean(unnormalized_probabilities, axis=0)
# print("Average Probabilities:", avg_probabilities)

# # print the summary statistics
# # Print summary statistics
# print("\nSummary Statistics:")
# print(all_data.groupby('cluster_membership').agg({'prob_0': ['mean', 'std', 'min', 'max'], 'prob_1': ['mean', 'std', 'min', 'max']}))



# # # Create a scatter plot of log-weighted probabilities
# # plt.figure(figsize=(10, 8))

# # # Plot points for cluster 0 log probabilities normalized
# # cluster_0 = all_data[all_data['cluster_membership'] == 0]
# # plt.scatter(normalized_log_probs[all_data['cluster_membership'] == 0, 1], normalized_log_probs[all_data['cluster_membership'] == 0, 0], c='blue', alpha=0.5, s= 10, label='Cluster 0')

# # # Plot points for cluster 1 log probabilities normalized
# # cluster_1 = all_data[all_data['cluster_membership'] == 1]
# # plt.scatter(normalized_log_probs[all_data['cluster_membership'] == 1, 1], normalized_log_probs[all_data['cluster_membership'] == 1, 0], c='red', alpha=0.5, s= 10, label='Cluster 1')

# # # Set labels and title
# # plt.xlabel('Log Probability of belonging to Cluster 1')
# # plt.ylabel('Log Probability of belonging to Cluster 0')
# # plt.title('Cluster Membership Log-Weighted Probabilities')
# # plt.legend()
# # plt.grid(True, linestyle='--', alpha=0.7)
# # plt.savefig(os.path.join(clusterings_results_path, 'cluster_membership_log_weighted_probabilities.png'))
# # plt.show()




# # Print summary statistics for log-weighted probabilities
# print("\nSummary Statistics for Log-Weighted Probabilities:")
# print(all_data.groupby('cluster_membership').agg({'log_prob_0': ['mean', 'std', 'min', 'max'], 'log_prob_1': ['mean', 'std', 'min', 'max']}))

# # # Visualize the distribution of probabilities
# # plt.figure(figsize=(10, 6))
# # plt.hist(probabilities[:, 0], bins=20, alpha=0.5, label='Cluster 0')
# # plt.hist(probabilities[:, 1], bins=20, alpha=0.5, label='Cluster 1')
# # plt.xlabel('Probability')
# plt.ylabel('Number of Samples')
# plt.title('Probability Distribution for Clusters')
# plt.legend(loc='upper right')
# # plt.show()
# plt.savefig(os.path.join(clusterings_results_path, 'probability_distribution.png'))
# plt.close()

# # define well clustered calls as those with probability above 0.75 for at least one cluster

# well_clustered_calls_cluster_0 = all_data[all_data['prob_0'] > 0.75]
# well_clustered_calls_cluster_1 = all_data[all_data['prob_1'] > 0.75]

# middle_calls = all_data[(all_data['prob_0'] < 0.60) & (all_data['prob_1'] < 0.60)]


# # print the number of well-clustered calls
# print(f"Number of well-clustered calls in Cluster 0: {len(well_clustered_calls_cluster_0)}")
# print(f"Number of well-clustered calls in Cluster 1: {len(well_clustered_calls_cluster_1)}")

# # print the number of calls in the middle
# print(f"Number of calls in the middle: {len(middle_calls)}")
# # print IDs middle calls
# print(middle_calls['call_id'].head())

# # take onsets and offsets of the middle calls and print spectrograms
# middle_calls = middle_calls.sample(len(middle_calls))
# plot_and_save_extreme_calls(middle_calls, audio_path, clusterings_results_path, 'middle_calls')
