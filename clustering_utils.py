import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy.io import wavfile
import librosa as lb
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import skfuzzy as fuzz
from math import pi
import seaborn as sns
import itertools
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler






def plot_dendrogram(model, num_clusters=None, **kwargs):
    """
    Plot a dendrogram for hierarchical clustering.

    Args:
        model: The hierarchical clustering model (e.g., from scikit-learn).
        num_clusters (int): The number of clusters desired. If provided, a threshold line will be drawn on the dendrogram to indicate where to cut it.
        **kwargs: Additional keyword arguments to pass to the dendrogram function.

    Returns:
        threshold (float): The distance threshold at which to cut the dendrogram.
        linkage_matrix (numpy.ndarray): The linkage matrix used to construct the dendrogram.
        counts (numpy.ndarray): Counts of samples under each node in the dendrogram.
        n_samples (int): Total number of samples.
        labels (numpy.ndarray): Labels assigned to each sample by the clustering model.
    """
    
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])  # Initialize counts of samples under each node
    n_samples = len(model.labels_)  # Total number of samples
    
    # Iterate over merges to calculate counts
    for i, merge in enumerate(model.children_):
        current_count = 0
        # Iterate over children of merge
        for child_idx in merge:
            if (child_idx < n_samples):
                current_count += 1  # Leaf node
            else:
                current_count += counts[child_idx - n_samples]  # Non-leaf node
        counts[i] = current_count  # Update counts
        
    # Construct the linkage matrix
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    
    # Debug: stampa le dimensioni degli array
    print(f"linkage_matrix shape: {linkage_matrix.shape}")
    print(f"linkage_matrix: {linkage_matrix}")

    # Plot the dendrogram
    dendrogram(linkage_matrix, **kwargs, leaf_rotation= 90. , leaf_font_size=5.0, truncate_mode='level', p=4, above_threshold_color='gray')
    #     p : int, optional
    #     The ``p`` parameter for ``truncate_mode``.
    # truncate_mode : str, optional
    #     The dendrogram can be hard to read when the original
    #     observation matrix from which the linkage is derived is
    #     large. Truncation is used to condense the dendrogram. There
    #     are several modes:

    #     ``None``
    #       No truncation is performed (default).
    #       Note: ``'none'`` is an alias for ``None`` that's kept for
    #       backward compatibility.
   
    # Plot the threshold line if num_clusters is specified
    if num_clusters is not None:
        max_d = np.max(model.distances_)
        threshold = max_d / (num_clusters - 1)
        plt.axhline(y=threshold, color='crimson', linestyle='--', label=f'{num_clusters} clusters', linewidth=1.5)
        plt.legend()
    
    plt.xlabel('Sample index or Cluster size', fontsize=10, fontfamily='Palatino Linotype')
    plt.ylabel('Distance', fontsize=10, fontfamily='Palatino Linotype')
    plt.title('Hierarchical Clustering Dendrogram', fontsize=13, fontfamily='Palatino Linotype')
    plt.show()
    plt.savefig('hierarchical_clustering_dendrogram.png')

    return threshold, linkage_matrix, counts, n_samples, model.labels_
    














# Define the function to find the elbow point
def find_elbow_point(scores):
    n_points = len(scores)
    all_coord = np.vstack((range(n_points), scores)).T
    first_point = all_coord[0]
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coord - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    best_index = np.argmax(dist_to_line)
    return best_index + 2  





# Function to get 5 random samples for each cluster
def get_random_samples(df, cluster_col, num_samples=5):
    random_samples = {}
    for cluster in df[cluster_col].unique():
        cluster_df = df[df[cluster_col] == cluster]
        if len(cluster_df) >= num_samples:
            random_samples[cluster] = cluster_df.sample(num_samples)
        else:
            random_samples[cluster] = cluster_df
    return random_samples













def segment_spectrogram(spectrogram, onsets, offsets, sr=44100):
    # Initialize lists to store spectrogram slices
    calls_S = []
    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Convert time (in seconds) to sample indices
        onset_frames = lb.time_to_frames(onset, sr=sr)
        offset_frames = lb.time_to_frames(offset, sr=sr)

        call_spec = spectrogram[:, onset_frames: offset_frames]

        # Append the scaled log-spectrogram slice to the calls list
        calls_S.append(call_spec)
    
    return calls_S



  

# Function to extract and plot audio segments
def plot_audio_segments(samples_dict, audio_path, clusterings_results_path, cluster_membership_label):
    for cluster, samples in samples_dict.items():
        fig, axes = plt.subplots(1, len(samples), figsize=(2 * len(samples), 2))
        fig.suptitle(f'Cluster {cluster} Audio Segments')
        if len(samples) == 1:
            axes = [axes]

        for idx, (i, sample) in enumerate(samples.iterrows()):
            audio_file = os.path.join(audio_path, sample['recording'] + '.wav')
            if os.path.exists(audio_file):
                # Load the audio file with librosa
                data, sr = lb.load(audio_file, sr=44100)
                
                # Compute the mel spectrogram
                S = lb.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmin=2000, fmax=10000)
                log_S = lb.power_to_db(S, ref=np.max)

                # Segment the spectrogram
                calls_S = segment_spectrogram(log_S, [sample['onsets_sec']], [sample['offsets_sec']], sr=sr)
                call_S = calls_S[0]

                # Convert onset seconds with decimals to readable format
                # onset_sec = sample['onsets_sec']
                # if onset_sec < 60:
                #     onset_time = f"{onset_sec:.2f} sec"
                # else:
                #     minutes = int(onset_sec // 60)
                #     seconds = onset_sec % 60
                #     onset_time = f"{minutes} min & {seconds:.2f} sec"

                # Plot the audio segment
                img= axes[idx].imshow(call_S, aspect='auto', origin='lower', cmap='magma')
                axes[idx].set_title(f'Call {idx + 1} of {sample["recording"]} \n cluster {cluster}', fontsize=6)

                axes[idx].set_xlabel('Time', fontsize=5)
                axes[idx].set_ylabel('Frequency', fontsize=5)
                fig.colorbar(img, ax=axes[idx])
            else:
                print(f'Audio file {audio_file} not found')

        # Save the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f'cluster_{cluster}_{cluster_membership_label}.png'
        plt.savefig(os.path.join(clusterings_results_path, plot_filename))

  




# # Define a function to plot UMAP with clusters
# def plot_umap_with_clusters(ax, embedding, labels, n_clusters, title):
#     ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=10, cmap='Spectral')
#     ax.set_title(f'UMAP with {n_clusters} clusters')
#     ax.set_xlabel('UMAP Dimension 1')
#     ax.set_ylabel('UMAP Dimension 2')

# # Find elbow points
# elbow_points = {
#     "BIC": best_elbow_n_clusters_bic,
#     "Silhouette": best_elbow_n_clusters_silhouette,
#     "Calinski": best_elbow_n_clusters_calinski
# }

# # Create subplots
# fig, axs = plt.subplots(1, len(elbow_points), figsize=(20, 5))

# # Plot the best model for each score
# for i, (score_name, n_clusters) in enumerate(elbow_points.items()):
#     gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(data_scaled)
#     labels = gmm.predict(data_scaled)
#     plot_umap_with_clusters(axs[i], standard_embedding, labels, n_clusters, score_name)

# plt.tight_layout()
# plt.show()



def silhouette_visualizer(data, n_clusters, title, method='gmm', **kwargs):
    """
    Visualizza il coefficiente silhouette per vari metodi di clustering.
    
    Args:
        data (array-like): Dati da clusterizzare.
        n_clusters (int): Numero di cluster desiderati.
        title (str): Titolo del grafico.
        method (str): Metodo di clustering da usare ('fcm', 'gmm', 'dbscan', 'agglomerative').
        **kwargs: Parametri aggiuntivi per il metodo di clustering scelto.
    """
    # Esegui il clustering in base al metodo specificato
    if method == 'fcm':
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
        )
        cluster_labels = np.argmax(u, axis=0)
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, **kwargs)
        cluster_labels = model.fit_predict(data)
    elif method == 'dbscan':
        model = DBSCAN(**kwargs)
        cluster_labels = model.fit_predict(data)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        cluster_labels = model.fit_predict(data)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Calcola il coefficiente silhouette
    silhouette_avg = silhouette_score(data, cluster_labels)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)
    
    print(f"Average silhouette score: {silhouette_avg}")

    # Crea il grafico del coefficiente silhouette
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(9, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # Usa colori più chiari
        color = sns.color_palette("pastel", n_colors=n_clusters)[i]
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title(title)
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))

    plt.tight_layout()
    plt.show()

# Esempio d'uso:
# silhouette_visualizer(data_scaled, n_clusters=3, title='Silhouette Plot', method='gmm', covariance_type='full', random_state=42)

import os
import matplotlib.pyplot as plt
import pandas as pd

def statistical_report(all_data, cluster_membership, n_clusters, metadata, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Add cluster membership to all_data
    all_data['cluster_membership'] = cluster_membership

    # Drop specified columns
    all_data = all_data.drop(['recording', 'Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)
    
    # Group by cluster membership and calculate the mean of each feature
    statistical_report_df = all_data.groupby('cluster_membership').mean().reset_index()
    
    # Add the number of samples in each cluster
    n_samples = all_data['cluster_membership'].value_counts().sort_index()
    statistical_report_df['n_samples'] = n_samples

    # Save the statistical report to a CSV file
    csv_file_path = os.path.join(output_folder, 'statistical_report.csv')
    statistical_report_df.to_csv(csv_file_path, index=False)

    # Convert and export the statistical report to LaTeX
    latex_file_path = os.path.join(output_folder, 'statistical_report.tex')
    statistical_report_df.to_latex(latex_file_path, index=False)

    # Colors for clusters
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    color_map = {i: color for i, color in enumerate(colors[:n_clusters])}
    color_map[-1] = 'slategray'  # Color for the noise cluster

    # Separate plots for groups of features
    features = list(all_data.columns[:-1])  # Exclude 'cluster_membership'
    num_features = len(features)
    features_per_plot = 4  # Number of features per plot
    num_plots = (num_features + features_per_plot - 1) // features_per_plot  # Calculate the number of plots needed

    for i in range(num_plots):
        start_idx = i * features_per_plot
        end_idx = min(start_idx + features_per_plot, num_features)
        plot_features = features[start_idx:end_idx]

        fig, axs = plt.subplots(1, len(plot_features), figsize=(20, 5))

        # Ensure axs is a list even if it contains a single subplot
        if len(plot_features) == 1:
            axs = [axs]

        for j, feature in enumerate(plot_features):
            data = [all_data[all_data['cluster_membership'] == k][feature] for k in range(-1, n_clusters)]

            bplot = axs[j].boxplot(data, patch_artist=True, showfliers=False)
            for patch, k in zip(bplot['boxes'], range(-1, n_clusters)):
                patch.set_alpha(0.1)
                patch.set_facecolor(color_map.get(k, 'slategray'))

            # Overlay scatterplot
            for cluster in range(-1, n_clusters):
                cluster_data = all_data[all_data['cluster_membership'] == cluster]
                axs[j].scatter([cluster + 2] * len(cluster_data), cluster_data[feature], 
                               alpha=0.3, c=color_map.get(cluster, 'slategray'), edgecolor=color_map.get(cluster, 'slategray'), s=13, label=f'Cluster {cluster}')
                
                axs[j].set_title(f'{feature} per cluster', size=7)
                axs[j].set_xlabel('Cluster', size=7)
                axs[j].set_ylabel('Value', size=7)

            # Set x-tick labels to show cluster numbers including -1
            axs[j].set_xticks(range(1, n_clusters + 2))
            axs[j].set_xticklabels(['-1'] + [str(i) for i in range(n_clusters)])

        plt.tight_layout()

        # Save the plot to a file
        plot_file_path = os.path.join(output_folder, f'statistical_report_part_{i+1}.png')
        plt.savefig(plot_file_path)
        # plt.show()

    return statistical_report_df






# # Funzione per creare e salvare il report statistico con boxplot e scatterplot sovrapposti
# def statistical_report(all_data, cluster_membership,n_clusters, metadata, output_folder):
#     # Crea un DataFrame per memorizzare il report statistico
#     all_data['cluster_membership'] = cluster_membership

#     all_data =  all_data.drop(['recording','Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)
    
#     # Raggruppa per appartenenza al cluster e calcola la media di ogni caratteristica
#     statistical_report_df = all_data.groupby('cluster_membership').mean().reset_index()
    
    
#     # Aggiungi il numero di campioni in ogni cluster
#     n_samples = all_data['cluster_membership'].value_counts().sort_index()
#     statistical_report_df['n_samples'] = n_samples

#     # Salva il report statistico su file CSV
#     csv_file_path = os.path.join(output_folder, 'statistical_report.csv')
#     statistical_report_df.to_csv(csv_file_path, index=False)

#     # Converti ed esporta il report statistico in LaTeX
#     latex_file_path = os.path.join(output_folder, 'statistical_report.tex')
#     statistical_report_df.to_latex(latex_file_path, index=False)

#     # Colori accessibili per i boxplot e scatterplot
#     colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
#     color_map = {i: color for i, color in enumerate(colors[:n_clusters])}
#     color_map[-1] = 'maroon'  # Colore per il cluster di rumore
#     # Plot separati per gruppi di caratteristiche
#     features = list(all_data.columns[:-1])  # Escludi 'cluster_membership'
#     num_features = len(features)
#     features_per_plot = 4  # Numero di caratteristiche per ogni plot
#     num_plots = (num_features + features_per_plot - 1) // features_per_plot  # Calcola il numero di plot necessari

#     for i in range( num_plots ):
#         start_idx = i * features_per_plot
#         end_idx = min(start_idx + features_per_plot, num_features)
#         plot_features = features[start_idx:end_idx]

#         fig, axs = plt.subplots(1, len(plot_features), figsize=(20, 5))

#         # Assicurati che axs sia una lista anche se contiene un solo subplot
#         if len(plot_features) == 1:
#             axs = [axs]

#         for j, feature in enumerate(plot_features):
#             data = [all_data[all_data['cluster_membership'] == k][feature] for k in range(n_clusters)]
            
#             bplot = axs[j].boxplot(data, patch_artist=True, showfliers= False)
#             for patch, k in zip(bplot['boxes'], range(-1, n_clusters - 1)):
#                 patch.set_alpha(0.1)
#                 patch.set_facecolor(color_map.get(k, 'maroon'))

                
#                 # Overlay scatterplot
#             for cluster in range(-1, n_clusters -1):
#                 cluster_data = all_data[all_data['cluster_membership'] == cluster]
#                 axs[j].scatter([cluster + 1] * len(cluster_data), cluster_data[feature], 
#                                alpha=0.3, c=color_map.get(cluster, 'maroon'), edgecolor=color_map.get(cluster, 'maroon'), s=13, label=f'Cluster {cluster}')
                
#                 axs[j].set_title(f'{feature} per cluster', size=7)
#                 axs[j].set_xlabel('Cluster', size=7)
#                 axs[j].set_ylabel('Value', size=7)

#         plt.tight_layout()

#         # Salva il plot su file
#         plot_file_path = os.path.join(output_folder, f'statistical_report_part_{i+1}.png')
#         plt.savefig(plot_file_path)
#         # plt.show()

#     return statistical_report_df


    # # To plot all features in a single plot
    # fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(40, 25))  # Creiamo una griglia di subplot 6x5 (per un totale di 30 spazi)

    # # Appiattiamo axs per un facile accesso
    # axs = axs.flatten()

    # for j, feature in enumerate(features):
    #     data = [all_data[all_data['cluster_membership'] == k][feature] for k in range(n_clusters)]
        
    #     bplot = axs[j].boxplot(data, patch_artist=True, showfliers= True)
    #     for patch, k in zip(bplot['boxes'], range(n_clusters)):
    #         patch.set_facecolor(color_map[k])
    #         patch.set_alpha(0.1)

    #         # Scatterplot sovrapposto
    #     for cluster in range(n_clusters):
    #         cluster_data = all_data[all_data['cluster_membership'] == cluster]
    #         axs[j].scatter([cluster + 1] * len(cluster_data), cluster_data[feature], 
    #                         alpha=0.3, c=color_map[cluster], edgecolor='k', s=13, label=f'Cluster {cluster}')
            
    #         axs[j].set_title(f'{feature}', size=7)
    #         axs[j].set_xlabel('Clusters', size=5)
    #         axs[j].set_ylabel('Value', size=5)

    
    # # Rimuove assi vuoti se ce ne sono meno di 30
    # for j in range(len(features), len(axs)):
    #     fig.delaxes(axs[j])

    # plt.tight_layout()

    # # Salva il plot su file
    # plot_file_path = os.path.join(output_folder, 'statistical_report_all_features.png')
    # plt.savefig(plot_file_path)
    # plt.show()

    # return statistical_report_df





def create_statistical_report_with_radar_plots(all_data, cluster_membership, n_clusters, metadata, output_folder):
  
    all_data = all_data.drop(['recording', 'Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)

    # Separare la colonna 'cluster_membership' dalle altre feature
    cluster_membership_col = all_data['cluster_membership']
    features = all_data.drop(['cluster_membership'], axis=1)

    # Applicare lo scaling alle feature
    scaled_features = StandardScaler().fit_transform(features)

    # Convertire l'array scalato in DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

    # Riaggiungere la colonna 'cluster_membership'
    scaled_features_df['cluster_membership'] = cluster_membership_col.values

    # Raggruppa per appartenenza al cluster e calcola la media di ogni caratteristica
    statistical_report_df = scaled_features_df.groupby('cluster_membership').mean().reset_index()

    # Aggiungi il numero di campioni in ogni cluster
    n_samples = scaled_features_df['cluster_membership'].value_counts().sort_index()
    statistical_report_df['n_samples'] = n_samples.values


    # Salva il report statistico su file CSV
    csv_file_path = os.path.join(output_folder, 'statistical_report.csv')
    statistical_report_df.to_csv(csv_file_path, index=False)

    # Converti ed esporta il report statistico in LaTeX
    latex_file_path = os.path.join(output_folder, 'statistical_report.tex')
    statistical_report_df.to_latex(latex_file_path, index=False)

    # Creare radar plot per visualizzare le variazioni delle feature per cluster
    features = list(all_data.columns[:-1])  # Escludi 'cluster_membership'
    num_features = len(features)
    num_clusters = len(statistical_report_df)

    # Definire colori distinti per ogni cluster
    colors = plt.cm.get_cmap('tab10', n_clusters)

    # Funzione per creare un singolo radar plot
    def create_radar_plot(data, title, color, ax):
        categories = list(data.keys())
        values = list(data.values())
        values += values[:1]  # Chiudi il cerchio

        angles = [n / float(num_features) * 2 * pi for n in range(num_features)]
        angles += angles[:1]

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='dimgrey', size=4.5, fontweight='bold')
        
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color= color, alpha=0.3)

        ax.set_title(title, size=10, color=color, y=1.1)

    # Creare una griglia di radar plot
    num_cols = 3
    num_rows = int(np.ceil(num_clusters / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), subplot_kw=dict(polar=True))

    axs = axs.flatten()

    for i, row in statistical_report_df.iterrows():
        data = row[features].to_dict()
        color = colors(i)
        create_radar_plot(data, f'Cluster {int(row["cluster_membership"])}', color, axs[i])

    for j in range(num_clusters, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plot_file_path = os.path.join(output_folder, 'radar_plots_clusters.png')
    plt.savefig(plot_file_path)
    # plt.show()

    return statistical_report_df


def radarplot_individual(all_data, output_folder):
    # Rimuovere colonne non necessarie
    all_data = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)

    # Applicare lo scaling alle feature
    scaled_features = StandardScaler().fit_transform(all_data.drop(['recording'], axis=1))

    # Convertire l'array scalato in DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=all_data.columns[:-1])

    # Riaggiungere la colonna 'recording'
    scaled_features_df['recording'] = all_data['recording'].values

    # Raggruppa per recording e calcola la media di ogni caratteristica
    statistical_report_df = scaled_features_df.groupby('recording').mean().reset_index()

    # Aggiungi il numero di campioni in ogni recording
    n_samples = scaled_features_df['recording'].value_counts().sort_index()
    statistical_report_df['n_samples'] = n_samples.values

    # Salva il report statistico su file CSV
    csv_file_path = os.path.join(output_folder, 'statistical_report.csv')
    statistical_report_df.to_csv(csv_file_path, index=False)

    # Converti ed esporta il report statistico in LaTeX
    latex_file_path = os.path.join(output_folder, 'statistical_report.tex')
    statistical_report_df.to_latex(latex_file_path, index=False)

    # Creare radar plot per visualizzare le variazioni delle feature per recording
    features = list(all_data.columns[:-1])  # Escludi 'recording'
    num_features = len(features)
    num_files = len(statistical_report_df)

    custom_colors = [
        "steelblue", "darkcyan", "mediumseagreen", 
        "indianred", "goldenrod", "orchid", 
        "lightskyblue", "limegreen", "tomato", 
        "mediumslateblue", "darkolivegreen", "cornflowerblue"
    ]

    # Funzione per creare un singolo radar plot
    def create_radar_plot(data, title, color, ax):
        categories = list(data.keys())
        values = list(data.values())
        values += values[:1]  # Chiudi il cerchio

        angles = [n / float(num_features) * 2 * pi for n in range(num_features)]
        angles += angles[:1]

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='dimgrey', size=5, fontweight='bold')
        
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.3)

        ax.set_title(title, size=12, color=color, y=1.1)

    # Creare una griglia di radar plot
    num_cols = 3
    num_rows = int(np.ceil(num_files / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 9, num_rows * 9), subplot_kw=dict(polar=True))

    axs = axs.flatten()

    for i, row in statistical_report_df.iterrows():
        data = row[features].to_dict()
        color = custom_colors[i % len(custom_colors)]
        
        # Controllo per gestire il numero variabile di plot
        if i < len(axs) and axs[i] is not None:
            create_radar_plot(data, f'Recording of {row["recording"]}', color, axs[i])

    # Rimuovere gli assi extra se non ci sono dati sufficienti per riempire la griglia
    for j in range(len(statistical_report_df), len(axs)):
        if axs[j] is not None:
            fig.delaxes(axs[j])

    plt.tight_layout()
    plot_file_path = os.path.join(output_folder, 'radar_plots_recordings.png')
    plt.savefig(plot_file_path)
    # plt.show()

    # Aggiungi la gestione per plottare 6 radar plot per griglia
    if num_files > 6:
        for k in range(2):
            fig, axs = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))
            axs = axs.flatten()
            
            start_idx = k * 6
            end_idx = min(start_idx + 6, num_files)
            
            for idx in range(start_idx, end_idx):
                data = statistical_report_df.iloc[idx][features].to_dict()
                color = custom_colors[idx % len(custom_colors)]
                
                create_radar_plot(data, f'Recording of {statistical_report_df.iloc[idx]["recording"]}', color, axs[idx - start_idx])
            
            plt.tight_layout()
            plot_file_path = os.path.join(output_folder, f'radar_plots_recordings_{k + 1}.png')
            plt.savefig(plot_file_path)

    return statistical_report_df
















