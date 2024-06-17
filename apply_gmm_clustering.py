
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

from sklearn.cluster import AgglomerativeClustering
from clustering_utils import get_random_samples, plot_audio_segments
from clustering_utils import get_random_samples, plot_audio_segments, statistical_report, create_statistical_report_with_radar_plots


# features_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/_results_high_quality_dataset_'
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_'

# metadata_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/High_quality_dataset/high_quality_dataset_metadata.csv'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'

audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset'

# Path to save the results
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_gmm_clustering_'
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

# # Plot the UMAP embeddings with the cluster membership
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# custom_colors = ['b', 'orange','LightCoral', 'MediumSlateBlue', 'ForestGreen', 'DarkTurquoise', 'DarkSlateBlue','LightSalmon', 'MediumPurple', 'LimeGreen', 'Cyan', 'Magenta', 'Yellow', 'LightCoral', 'MediumSlateBlue', 'ForestGreen', 'DarkTurquoise', 'DarkSlateBlue']

# for j in range(n_components):
#     ax.scatter(standard_embedding[cluster_membership == j, 0], 
#                standard_embedding[cluster_membership == j, 1], 
#                standard_embedding[cluster_membership == j, 2], c=custom_colors[j], s=10, label=f'Cluster {j}', alpha=0.3)
# ax.legend(loc='upper right', markerscale=2)

# plt.title(f'UMAP projection of the dataset with Gaussian Mixture Model clustering with {n_components} components')
# plt.legend()
# plt.show()

# # extract segments from the audio files
# random_samples = get_random_samples(all_data, 'cluster_membership', num_samples=5)
# print('Random samples selected')
# # # Plot the audio segments
# plot_audio_segments(random_samples, audio_path, clusterings_results_path, f'gaussian_mixture_{n_components}_membership')
# print('Audio segments plotted')

gmm.predict_proba(features_scaled)


# stats = statistical_report(all_data, cluster_membership, n_components, metadata, clusterings_results_path)
# print(stats)


radar= statistical_report_df = create_statistical_report_with_radar_plots(all_data, cluster_membership, n_components, metadata, clusterings_results_path)