import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap
from sklearn.manifold import TSNE
import shap
import seaborn as sns
from clustering_utils import radarplot_individual
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from clustering_utils import radarplot_individual, plot_dendrogram
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Definisci i percorsi dei file
file_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_without_jtfs'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
clustering_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\individual_clustering_results\\_without_jtfs__'
# Check if the directory exists, if not, create it
if not os.path.exists(clustering_results_path):
    os.makedirs(clustering_results_path)



def plot_pca(features_scaled, recording_encoded, label_encoder, clustering_results_path):
    pca = PCA(n_components=3)
    pca_transformed = pca.fit_transform(features_scaled)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(recording_encoded):
        ax.scatter(pca_transformed[recording_encoded == label, 0],
                   pca_transformed[recording_encoded == label, 1],
                   pca_transformed[recording_encoded == label, 2],
                   label=label_encoder.inverse_transform([label])[0],
                   alpha=0.3)
    ax.set_title('PCA: Projection of Features')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(clustering_results_path, 'pca_plot_3d.png'))
    plt.show()

def plot_umap(features_scaled, recording_encoded, label_encoder, clustering_results_path):
    umap_reducer = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7)
    umap_transformed = umap_reducer.fit_transform(features_scaled)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(recording_encoded):
        ax.scatter(umap_transformed[recording_encoded == label, 0],
                   umap_transformed[recording_encoded == label, 1],
                   umap_transformed[recording_encoded == label, 2],
                   label=label_encoder.inverse_transform([label])[0],
                   alpha=0.3)
    ax.set_title('UMAP: Projection of Features')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(clustering_results_path, 'umap_plot_3d.png'))
    plt.show()

def plot_lda(features_scaled, recording_encoded, label_encoder, clustering_results_path):
    lda = LDA(n_components=2)
    lda_transformed = lda.fit(features_scaled, recording_encoded).transform(features_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in np.unique(recording_encoded):
        ax.scatter(lda_transformed[recording_encoded == label, 0],
                   lda_transformed[recording_encoded == label, 1],
                   label=label_encoder.inverse_transform([label])[0],
                   alpha=0.6)
    ax.set_title('LDA: Projection of Features')
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(clustering_results_path, 'lda_plot.png'))
    plt.show()

def plot_tsne(features_scaled, recording_encoded, label_encoder, clustering_results_path):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in np.unique(recording_encoded):
        ax.scatter(tsne_results[recording_encoded == label, 0],
                   tsne_results[recording_encoded == label, 1],
                   label=label_encoder.inverse_transform([label])[0],
                   alpha=0.6)
    ax.set_title('t-SNE: Projection of Features')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(clustering_results_path, 'tsne_plot.png'))
    plt.show()


def feature_importance_shap(features_scaled, lda, clustering_results_path):
    explainer = shap.KernelExplainer(lda.predict, features_scaled)
    shap_values = explainer.shap_values(features_scaled[np.random.choice(features_scaled.shape[0], 100, replace=False)])

    shap.summary_plot(shap_values, features_scaled, feature_names=all_data.columns.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id']))
    plt.savefig(os.path.join(clustering_results_path, 'shap_summary_plot.png'))
    plt.show()

def statistical_analysis(all_data, recording_encoded, clustering_results_path):
    results = []

    feature_names = all_data.columns.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording'])
    for feature in feature_names:
        feature_data = all_data[feature]
        groups = [feature_data[recording_encoded == label] for label in np.unique(recording_encoded)]
        if len(groups) >= 2:
            stat, p_value = ttest_ind(groups[0], groups[1])  # Confronta le prime due classi
            results.append((feature, stat, p_value))

            print(f'T-Test per {feature}: Statistica={stat}, p-value={p_value}')
            with open(os.path.join(clustering_results_path, 'statistical_analysis.txt'), 'a') as f:
                f.write(f'T-Test per {feature}: Statistica={stat}, p-value={p_value}\n')

    # Salva i risultati in un file CSV
    results_df = pd.DataFrame(results, columns=['Feature', 'Statistic', 'p-value'])
    results_df.to_csv(os.path.join(clustering_results_path, 'statistical_analysis_results.csv'), index=False)

    # Visualizza i risultati con un grafico a barre dei p-value
    plt.figure(figsize=(12, 8))
    plt.bar(results_df['Feature'], results_df['p-value'], color='blue')
    plt.axhline(y=0.05, color='r', linestyle='--')  # Linea di soglia per significatività statistica
    plt.xlabel('Features')
    plt.ylabel('p-value')
    plt.xticks(rotation=90)
    plt.title('p-value per ciascuna caratteristica')
    plt.tight_layout()
    plt.savefig(os.path.join(clustering_results_path, 'p_value_barplot.png'))
    plt.show()



def evaluate_gmm_clustering(features_scaled, n_max_components=11, covarience_type='full', n_init=5, max_iter=1000, toll=1e-3, random_state=42, init_params='kmeans'):
    components = range(2, n_max_components)
    
    # Dictionary to store evaluation metrics for each number of components
    gmm_cluster_evaluation_per_number_clusters = {
        n_comp: {'silhouette_score': 0, 'calinski_harabasz_score': 0, 'wcss': 999, 'bic': 999, 'aic': 999, 'number_of_clusters': 0} for n_comp in components
    }
    
    for n_comp in components:
        gmm = GaussianMixture(n_components=n_comp, covariance_type=covarience_type, n_init=n_init, max_iter=max_iter, tol=toll, random_state=random_state, init_params=init_params) 
        cluster_membership = gmm.fit_predict(features_scaled)
        
        number_of_clusters = len(set(cluster_membership))
        # Save the number of clusters
        gmm_cluster_evaluation_per_number_clusters[n_comp]['number_of_clusters'] = number_of_clusters
        
        # Compute the centroid of each cluster (using GMM's means_)
        cluster_centers = gmm.means_
        # Compute WCSS
        wcss = np.sum((np.linalg.norm(features_scaled - cluster_centers[cluster_membership], axis=1) ** 2))
        
        gmm_cluster_evaluation_per_number_clusters[n_comp]['silhouette_score'] = silhouette_score(features_scaled, cluster_membership)
        gmm_cluster_evaluation_per_number_clusters[n_comp]['calinski_harabasz_score'] = calinski_harabasz_score(features_scaled, cluster_membership)
        gmm_cluster_evaluation_per_number_clusters[n_comp]['wcss'] = wcss
        gmm_cluster_evaluation_per_number_clusters[n_comp]['bic'] = gmm.bic(features_scaled)
        gmm_cluster_evaluation_per_number_clusters[n_comp]['aic'] = gmm.aic(features_scaled)

        print('Number of clusters:', number_of_clusters) 
        print('Silhouette score:', gmm_cluster_evaluation_per_number_clusters[n_comp]['silhouette_score'])
        print('Calinski Harabasz score:', gmm_cluster_evaluation_per_number_clusters[n_comp]['calinski_harabasz_score'])
        print('WCSS:', gmm_cluster_evaluation_per_number_clusters[n_comp]['wcss'])    
        print('BIC:', gmm_cluster_evaluation_per_number_clusters[n_comp]['bic'])
        print('AIC:', gmm_cluster_evaluation_per_number_clusters[n_comp]['aic'])
        print()

    # Find the vector of silhouette scores, calinski harabasz scores, wcss, bic, and aic across the number of clusters
    wcss_vector_across_n_clusters = [result['wcss'] for result in gmm_cluster_evaluation_per_number_clusters.values()]
    bic_vector_across_n_clusters = [result['bic'] for result in gmm_cluster_evaluation_per_number_clusters.values()]
    aic_vector_across_n_clusters = [result['aic'] for result in gmm_cluster_evaluation_per_number_clusters.values()]
    
    # Find the elbow points
    wcss_elbow = KneeLocator(range(2, len(wcss_vector_across_n_clusters) + 2), wcss_vector_across_n_clusters, curve='convex', direction='decreasing', interp_method='interp1d', online=False)
    bic_elbow = KneeLocator(range(2, len(bic_vector_across_n_clusters) + 2), bic_vector_across_n_clusters, curve='convex', direction='decreasing', interp_method='polynomial', online=False)
    aic_elbow = KneeLocator(range(2, len(aic_vector_across_n_clusters) + 2), aic_vector_across_n_clusters, curve='convex', direction='decreasing', interp_method='polynomial', online=False)
    
    best_n_clusters_by_wcss_elbow = wcss_elbow.elbow
    best_n_clusters_by_bic_elbow = bic_elbow.elbow
    best_n_clusters_by_aic_elbow = aic_elbow.elbow
    
    results = {
        'gmm_cluster_evaluation': gmm_cluster_evaluation_per_number_clusters,
        'wcss_elbow': best_n_clusters_by_wcss_elbow,
        'bic_elbow': best_n_clusters_by_bic_elbow,
        'aic_elbow': best_n_clusters_by_aic_elbow
    }
    
    return results



def evaluate_hierarchical_clustering(features_scaled, n_max_clusters=11, linkage='ward', clusterings_results_path=clustering_results_path):
    hierarchical_cluster_evaluation_per_number_clusters = {
        n_clusters: {'silhouette_score': 0, 'calinski_harabasz_score': 0, 'wcss': 9999} for n_clusters in range(2, n_max_clusters)
    }
    
    for n_clusters in range(2, n_max_clusters):
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        cluster_membership = agg.fit_predict(features_scaled)
        
        centroids = np.array([features_scaled[cluster_membership == c].mean(axis=0) for c in range(n_clusters)])
        
        hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['silhouette_score'] = silhouette_score(features_scaled, cluster_membership)
        hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['calinski_harabasz_score'] = calinski_harabasz_score(features_scaled, cluster_membership)
        hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] = np.sum((np.linalg.norm(features_scaled - centroids[cluster_membership], axis=1) ** 2))
        
        print('Number of clusters:', n_clusters)
        print('Silhouette score:', hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['silhouette_score'])
        print('Calinski Harabasz score:', hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['calinski_harabasz_score'])
        print('WCSS:', hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['wcss'])
        print()

    # Save results
    hierarchical_cluster_evaluation_per_number_clusters_df = pd.DataFrame(hierarchical_cluster_evaluation_per_number_clusters).T
    hierarchical_cluster_evaluation_per_number_clusters_df.to_csv(os.path.join(clusterings_results_path, 'hierarchical_cluster_evaluation_per_number_clusters.csv'))
    hierarchical_cluster_evaluation_per_number_clusters_df.to_latex(os.path.join(clusterings_results_path, 'hierarchical_cluster_evaluation_per_number_clusters.tex'))
    
    # Find the optimal number of clusters
    wcss_vector_across_n_clusters = [hierarchical_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] for n_clusters in range(2, n_max_clusters)]
    wcss_elbow = KneeLocator(range(2, n_max_clusters), wcss_vector_across_n_clusters, curve='convex', direction='decreasing', interp_method='polynomial', online=False)
    best_n_clusters_by_wcss_elbow = wcss_elbow.elbow
    
    results = {
        'hierarchical_cluster_evaluation': hierarchical_cluster_evaluation_per_number_clusters,
        'wcss_elbow': best_n_clusters_by_wcss_elbow
    }
    
    return results, hierarchical_cluster_evaluation_per_number_clusters_df, best_n_clusters_by_wcss_elbow





if __name__ == '__main__':

        
    custom_colors = [
    "steelblue", "darkcyan", "mediumseagreen", 
    "indianred", "goldenrod", "orchid", 
    "lightskyblue", "limegreen", "tomato", 
    "mediumslateblue", "darkolivegreen", "cornflowerblue"
    ]

    # Leggi il metadata
    metadata = pd.read_csv(metadata_path)

    # Ottieni una lista di tutti i file CSV nella directory
    list_files = glob.glob(os.path.join(file_path, '*.csv'))

    # Concatena tutti i dati da tutti i file CSV in un unico DataFrame
    all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)

    # Rimuovi le righe con valori nulli
    all_data = all_data.dropna()

    # Standardizza le features
    scaler = StandardScaler()
    features = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id'], axis=1)  # Rimuovi colonne non numeriche
    features_scaled = scaler.fit_transform(features)

 
    

    # Codifica la variabile categorica 'recording' con LabelEncoder
    label_encoder = LabelEncoder()
    recording_encoded = label_encoder.fit_transform(all_data['recording'])

    unique_labels = np.unique(recording_encoded)
    
    
    
    plot_pca(features_scaled, recording_encoded, label_encoder, clustering_results_path)
    plot_umap(features_scaled, recording_encoded, label_encoder, clustering_results_path)
    plot_lda(features_scaled, recording_encoded, label_encoder, clustering_results_path)
    plot_tsne(features_scaled, recording_encoded, label_encoder, clustering_results_path)

    radarplot_individual(all_data, output_folder=clustering_results_path)
    

    # results = {}
    # elbow_results = pd.DataFrame(columns=['recording', 'elbow_wcss', 'elbow_bic', 'elbow_aic'])
    # unique_labels = np.unique(recording_encoded)

    # for label in unique_labels:
    #     features_label = features_scaled[recording_encoded == label]
    #     evaluation_results = evaluate_gmm_clustering(features_label)
    #     results[label] = evaluation_results

    #     clustering_results = {k: v for k, v in results[label].items() if isinstance(v, dict)}
    #     elbow_results_dict = {k: v for k, v in results[label].items() if not isinstance(v, dict)}

    #     clustering_results_df = pd.DataFrame.from_dict(clustering_results, orient='index')

    #     expected_columns = ['silhouette_score', 'calinski_harabasz_score', 'wcss', 'bic', 'aic', 'number_of_clusters']
    #     if clustering_results_df.shape[1] == len(expected_columns):
    #         clustering_results_df.columns = expected_columns
    #         clustering_results_df.to_csv(os.path.join(clustering_results_path, f'gmm_cluster_evaluation_per_number_clusters_{label}.csv'), index=True)
    #         with open(os.path.join(clustering_results_path, f'gmm_cluster_evaluation_per_number_clusters_{label}.tex'), 'w') as f:
    #             f.write(clustering_results_df.to_latex(index=True))

    #         elbow_results = elbow_results.append({
    #             'recording': label, 
    #             'elbow_wcss': elbow_results_dict['wcss_elbow'], 
    #             'elbow_bic': elbow_results_dict['bic_elbow'], 
    #             'elbow_aic': elbow_results_dict['aic_elbow']
    #         }, ignore_index=True)

    #     elbow_results.to_csv(os.path.join(clustering_results_path, 'elbow_results.csv'), index=False)
    #     elbow_results.to_latex(os.path.join(clustering_results_path, 'elbow_results.tex'))
        
    # Remove the 'continue' statement


    # for label in unique_labels:
    #     features_label = features_scaled[recording_encoded == label]
    #     results[label] = evaluate_hierarchical_clustering(features_label)

    #     # Crea un DataFrame dai risultati
    #     hierarchical_cluster_evaluation_per_number_clusters_df = pd.DataFrame.from_dict(results[label], orient='index')

    #     # Aggiungi i nomi delle colonne
    #     hierarchical_cluster_evaluation_per_number_clusters_df.columns = [
    #         'silhouette_score', 'calinski_harabasz_score', 'wcss'
    #     ]

    #     # Salva i risultati del clustering in un file CSV
    #     hierarchical_cluster_evaluation_per_number_clusters_df.to_csv(os.path.join(clustering_results_path, f'hierarchical_cluster_evaluation_per_number_clusters_{label}.csv'), index=True)

    #     # Salva i risultati del clustering in un file LaTeX
    #     with open(os.path.join(clustering_results_path, f'hierarchical_cluster_evaluation_per_number_clusters_{label}.tex'), 'w') as f:
    #         f.write(hierarchical_cluster_evaluation_per_number_clusters_df.to_latex(index=True))

    #     print(f'Hierarchical clustering done for label {label}')
    # print('Hierarchical clustering done')
