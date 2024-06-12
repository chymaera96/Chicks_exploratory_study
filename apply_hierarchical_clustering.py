
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

from sklearn.cluster import AgglomerativeClustering
from clustering_utils import get_random_samples, plot_audio_segments, plot_dendrogram
from clustering_utils import statistical_report, create_statistical_report_with_radar_plots

# features_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/_results_high_quality_dataset_'
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_'

# metadata_path = '/Users/ines/Dropbox/QMUL/BBSRC-chickWelfare/High_quality_dataset/high_quality_dataset_metadata.csv'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'

audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset'

# Path to save the results
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_hierarchical_clustering_'
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
features = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec','recording', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)


n_clusters = 5

agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', compute_distances=True)

cluster_membership = agg.fit_predict(features_scaled)

all_data['cluster_membership'] = cluster_membership

all_data.to_csv(os.path.join(clusterings_results_path, f'hierarchical_clustering__{n_clusters}_membership.csv'), index=False)
linkage_matrix = linkage(features_scaled, method='ward')


# Plot the dendrogram and get the cluster memberships
membership = plot_dendrogram(agg, num_clusters=n_clusters)
if membership is not None:
    print(membership)

# Get 5 random samples for each cluster
random_samples = get_random_samples(all_data, 'cluster_membership', num_samples=5)

# Plot the audio segments
plot_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')

# Get the statistical report

stats = statistical_report(all_data, cluster_membership,n_clusters, metadata, clusterings_results_path)
print(stats)


radar= statistical_report_df = create_statistical_report_with_radar_plots(all_data, cluster_membership, n_clusters, metadata, clusterings_results_path)