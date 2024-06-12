import os
import glob
import pandas as pd
import numpy as np
import umap.umap_ as umap
from gap_statistic import OptimalK
import matplotlib.pyplot as plt
from kneed import KneeLocator
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import seaborn as sns

# Percorsi dei file
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_silhouette_performance'

# Assicurati che la cartella esista
if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)

# Ottieni una lista di tutti i file CSV nella directory
list_files = glob.glob(os.path.join(features_path, '*.csv'))

# Leggi tutti i file CSV e concatenali in un singolo DataFrame
all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)

# Rimuovi i valori NaN
all_data = all_data.dropna()

# Scala i dati con StandardScaler sulle feature utilizzate
scaler = StandardScaler()
features = all_data.drop(['recording', 'Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)

def silhouette_visualizer(ax, data, n_clusters, title, method='gmm', **kwargs):
    """
    Visualizza il coefficiente silhouette per vari metodi di clustering.
    
    Args:
        ax (Axes): L'asse su cui disegnare il grafico.
        data (array-like): Dati da clusterizzare.
        n_clusters (int): Numero di cluster desiderati.
        title (str): Titolo del grafico.
        method (str): Metodo di clustering da usare ('fcm', 'gmm', 'dbscan', 'agglomerative').
        **kwargs: Parametri aggiuntivi per il metodo di clustering scelto.
    """
    # create a dataframe to store the silhouette values for all the methods
    silhouette_avg_data = pd.DataFrame()
    

    if method == 'Fuzzy C-Means':
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)
        cluster_labels = np.argmax(u, axis=0)
    elif method == 'Gaussian Mixture Model':
        model = GaussianMixture(n_components=n_clusters, **kwargs)
        cluster_labels = model.fit_predict(data)
    elif method == 'DBSCAN':
        model = DBSCAN(**kwargs)
        cluster_labels = model.fit_predict(data)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    elif method == 'Agglomerative Hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        cluster_labels = model.fit_predict(data)
    else:
        raise ValueError(f"Unsupported method: {method}")

    silhouette_avg = silhouette_score(data, cluster_labels)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)
    
    print(f"Average silhouette score for {method} ({title}): {silhouette_avg}")

    #save and export the silhouette values in a csv file for all the methods
    silhouette_avg_data[f'silhouette_values for {method}'] = sample_silhouette_values
    silhouette_avg_data.to_csv(os.path.join(clusterings_results_path, f'silhouette_values_{method}.csv'), index=False)

    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = sns.color_palette("pastel", n_colors=n_clusters)[i]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=4)
        y_lower = y_upper + 10

    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Silhouette coefficient values", fontsize=6)
    ax.set_ylabel("Cluster label", fontsize=6)

    ax.axvline(x=silhouette_avg, color="crimson", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.1, 1.1, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=4)


def create_subplot_figure(parametri, method):
    n_rows = len(parametri) // 2 + len(parametri) % 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 1 * n_rows))
    axes = axes.flatten()
    for i, params in enumerate(parametri):
        if method == 'Gaussian Mixture Model':
            silhouette_visualizer(axes[i], features_scaled, params, f"GMM with {params} clusters", method='Gaussian Mixture Model')
        elif method == 'DBSCAN':
            eps, min_samples = params
            silhouette_visualizer(axes[i], features_scaled, None, f"DBSCAN with eps={eps}, min_samples={min_samples}", method='DBSCAN', eps=eps, min_samples=min_samples)
        elif method == 'Agglomerative Hierarchical':
            silhouette_visualizer(axes[i], features_scaled, params, f"Agglomerative with {params} clusters", method='Agglomerative Hierarchical')
        elif method == 'Fuzzy C-Means':
            silhouette_visualizer(axes[i], features_scaled, params, f"Fuzzy C-Means with {params} clusters", method='Fuzzy C-Means')

    # Rimuove gli assi vuoti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'Silhouette Analysis for {method}', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to include the suptitle
    plt.savefig('C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_silhouette_performance\\silhouette_analysis_' + method.replace(' ', '_').lower() + '.png')
    plt.show()



# Parametri di clustering da testare
parametri_gmm = [2, 3, 4, 5, 6, 7, 8, 9, 10]
parametri_dbscan =[(5.6, 2), (6.1, 2), (5.9, 2), (5.9, 3), (5.7, 2), (5.6, 2), (5.7, 3), (5.4, 2), (5.4, 3)]
parametri_agglo = [2, 3, 4, 5, 6, 7, 8, 9, 10]
parametri_fcm = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Crea figure separate per ogni tecnica
create_subplot_figure(parametri_gmm, method='Gaussian Mixture Model')
create_subplot_figure(parametri_dbscan, method='DBSCAN')
create_subplot_figure(parametri_agglo, method='Agglomerative Hierarchical')
create_subplot_figure(parametri_fcm, method='Fuzzy C-Means')



