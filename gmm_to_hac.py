import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from kneed import KneeLocator
from gap_statistic import OptimalK
import umap.umap_ as umap
import logging
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
# import yaml
from scipy.cluster.hierarchy import dendrogram
from clustering_utils import plot_dendrogram


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def load_config(config_path):
#     """Load configuration from YAML file."""
#     with open(config_path, 'r') as file:
#         return yaml.safe_load(file)

def load_and_preprocess_data(features_path, metadata_path):
    """Load and preprocess data from CSV files."""
    try:
        list_files = glob.glob(os.path.join(features_path, '*.csv'))
        all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
        metadata = pd.read_csv(metadata_path)
        all_data = all_data.dropna()
        return all_data, metadata
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def scale_features(data):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    features = data.drop(['recording', 'Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)
    return scaler.fit_transform(features)

def evaluate_gmm(n_comp, features_scaled):
    """Evaluate GMM for a given number of components."""
    gmm = GaussianMixture(n_components=n_comp, covariance_type='full', n_init=5, max_iter=1000, tol=1e-3, random_state=42)
    cluster_membership = gmm.fit_predict(features_scaled)
    number_of_clusters = len(set(cluster_membership))
    
    cluster_centers = gmm.means_
    wcss = np.sum((np.linalg.norm(features_scaled - cluster_centers[cluster_membership], axis=1) ** 2))
    
    return {
        'number_of_clusters': number_of_clusters,
        'silhouette_score': silhouette_score(features_scaled, cluster_membership),
        'calinski_harabasz_score': calinski_harabasz_score(features_scaled, cluster_membership),
        'wcss': wcss,
        'bic': gmm.bic(features_scaled),
        'aic': gmm.aic(features_scaled)
    }

def find_optimal_clusters(features_scaled, n_max_components):
    """Find optimal number of clusters using GMM and BIC."""
    components = range(2, n_max_components)
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(evaluate_gmm, components, [features_scaled]*len(components)), 
                            total=len(components), desc="Evaluating GMM"))
    
    gmm_cluster_evaluation = dict(zip(components, results))
    
    bic_vector = [result['bic'] for result in gmm_cluster_evaluation.values()]
    bic_elbow = KneeLocator(range(2, len(bic_vector)+2), bic_vector, curve='convex', direction='decreasing', interp_method='polynomial', online=False)
    
    return bic_elbow.elbow

def perform_hac(features_scaled, n_clusters):
    """Perform Hierarchical Agglomerative Clustering."""
    hac = AgglomerativeClustering(n_clusters=n_clusters)
    return hac.fit_predict(features_scaled)

def plot_umap_embedding(umap_embedding, labels, n_clusters, output_path):
    """Plot UMAP embedding with HAC clusters."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    custom_colors = ["darkorange", "turquoise", 'red', 'DarkGreen', 'LightCoral', 'MediumSlateBlue']

    for j in range(n_clusters):
        ax.scatter(umap_embedding[labels == j, 0],
                   umap_embedding[labels == j, 1],
                   umap_embedding[labels == j, 2],
                   color=custom_colors[j % len(custom_colors)], alpha=0.2, label=f'Cluster {j+1}', s=7)

    ax.set_title('UMAP Embedding with HAC Clusters')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    # Hard-code the paths and parameters
    features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_'
    metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
    clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_combined_clustering_gmm_ahc'
    n_max_components = 11

    os.makedirs(clusterings_results_path, exist_ok=True)
    
    all_data, metadata = load_and_preprocess_data(features_path, metadata_path)
    all_data.to_csv(os.path.join(clusterings_results_path, 'all_data.csv'), index=False)
    
    features_scaled = scale_features(all_data)
    
    best_n_clusters = find_optimal_clusters(features_scaled, n_max_components)
    logging.info(f"Optimal number of clusters: {best_n_clusters}")
    
    hac_labels = perform_hac(features_scaled, best_n_clusters)
    all_data['hac_cluster'] = hac_labels
    all_data.to_csv(os.path.join(clusterings_results_path, 'combined_clustering_results_hac.csv'), index=False)
    
    umap_reducer = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7)
    umap_embedding = umap_reducer.fit_transform(features_scaled)
    
    plot_umap_embedding(umap_embedding, hac_labels, best_n_clusters, 
                        os.path.join(clusterings_results_path, 'combined_clustering_hac_3d.png'))
    
    hac_model = AgglomerativeClustering(n_clusters=best_n_clusters, compute_distances=True)
    hac_model.fit(features_scaled)
    plot_dendrogram(hac_model, num_clusters=best_n_clusters)
    
    silhouette_avg = silhouette_score(features_scaled, hac_labels)
    calinski_harabasz_avg = calinski_harabasz_score(features_scaled, hac_labels)
    
    logging.info(f'Silhouette Score for HAC: {silhouette_avg}')
    logging.info(f'Calinski Harabasz Score for HAC: {calinski_harabasz_avg}')

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

