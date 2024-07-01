import os
import glob
import pandas as pd
import numpy as np
import umap.umap_ as umap
# import umap  #install umap-learn
import matplotlib.pyplot as plt
from kneed import KneeLocator

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score
from sklearn.metrics import calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture

from sklearn.cluster import DBSCAN
from clustering_utils import get_random_samples, plot_audio_segments
from clustering_utils import statistical_report, create_statistical_report_with_radar_plots, plot_and_save_audio_segments



# features_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/_results_high_quality_dataset_'
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_'

# metadata_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/High_quality_dataset/high_quality_dataset_metadata.csv'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'

audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset'

# Path to save the results
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_dbscan_clustering_'
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


epsilon = 5.9
min_samples = 2

# Compute the DBSCAN clustering
db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(features_scaled)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels))

# - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)

# plot_audio_segments(audio_segments, n_segments=5)
# Save the results
all_data['cluster_membership'] = labels
all_data.to_csv(os.path.join(clusterings_results_path, f'dbscan_cluster_membership_eps_{epsilon}_min_samples_{min_samples}.csv'), index=False)

# file_csv = f'dbscan_cluster_membership_eps_{epsilon}_min_samples_{min_samples}.csv'

# # Perform UMAP for obtaining the standard embeddings for the clustering techniques
# umap_reducer = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7)
# standard_embedding = umap_reducer.fit_transform(features_scaled)

# # Plot the UMAP embeddings with the cluster membership and noise points
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# custom_colors = ['Turquoise','Orange', 'MediumPurple', 'Darkred' ,'DarkKhaki', 'LightSkyBlue', 'LightGreen', 'LightPink', 'LightSteelBlue', 'LightSeaGreen', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightSlateGray', 'LightYellow']

# for j in range(n_clusters_):
#     ax.scatter(standard_embedding[labels == j, 0], 
#                standard_embedding[labels == j, 1], 
#                standard_embedding[labels == j, 2], 
#                c=custom_colors[j % len(custom_colors)], 
#                s=10, label=f'Cluster {j}', alpha=0.3)

# # Plot noise points
# ax.scatter(standard_embedding[labels == -1, 0], 
#            standard_embedding[labels == -1, 1], 
#            standard_embedding[labels == -1, 2], 
#            c='k', 
#            s=10, 
#            label='Noise', alpha=0.5)

# plt.title('UMAP projection of the dataset with DBSCAN clustering')
# ax.set_xlabel('UMAP 1')
# ax.set_ylabel('UMAP 2')
# ax.set_zlabel('UMAP 3')
# plt.legend()
# plt.show()
# plt.savefig(os.path.join(clusterings_results_path, f'dbscan_cluster_membership_eps_{epsilon}_min_samples_{min_samples}.png'))


# extract segments from the audio files
random_samples = get_random_samples(all_data, 'cluster_membership', num_samples=5)
print('Random samples selected')
# # # Plot the audio segments
# plot_audio_segments(random_samples, audio_path, clusterings_results_path, file_csv)
# print('Audio segments plotted')


# Plot the audio segments and save audio files
plot_and_save_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')


# # Get the statistical report

# stats = statistical_report(all_data, labels, n_clusters_ -1 , metadata, clusterings_results_path)
# print(stats)


# radar= statistical_report_df = create_statistical_report_with_radar_plots(all_data, labels, n_clusters_ -1 , metadata, clusterings_results_path)