import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import umap
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import shap

# Define the file paths
file_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
clustering_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering'

if not os.path.exists(clustering_results_path):
    os.makedirs(clustering_results_path)

# Read metadata
metadata = pd.read_csv(metadata_path)

# Get a list of all CSV files in the directory
list_files = glob.glob(os.path.join(file_path, '*.csv'))



all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)

all_data = all_data.dropna()

scaler = StandardScaler()
features = all_data.drop(['recording','Call Number', 'onsets_sec', 'offsets_sec'], axis=1)
features_scaled = scaler.fit_transform(features)


# Function to generate distinct colors
def distinct_colors(n):
    colors = plt.cm.get_cmap('tab10', n)
    return colors(range(n))










# Function to perform PCA analysis and plot
def perform_pca_analysis(features_scaled, clustering_results_path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i, component in enumerate(range(2, 11)):
        pca = PCA(n_components=component)
        principalComponents = pca.fit_transform(features_scaled)

        print(f'Explained variance ratio for {component} components: {pca.explained_variance_ratio_}')
        print(f'Total explained variance for {component} components: {np.sum(pca.explained_variance_ratio_)}')
        print('')

        ax = axes[i // 3, i % 3]
        colors = distinct_colors(component)
        for j in range(component):
            ax.scatter(principalComponents[:, 0], principalComponents[:, j], label=f'PC{j+1}', alpha=0.2, c=colors[j], s=7)
        ax.set_title(f'PCA with {component} components')
        ax.legend()
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(clustering_results_path, 'pca_analysis.png'))
    plt.show()

    return principalComponents








# Function to plot 3D PCA
def plot_3d_pca(principalComponents, clustering_results_path):
    if principalComponents.shape[1] >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        colors = distinct_colors(len(principalComponents))

        for i in range(len(principalComponents)):
            ax.scatter(principalComponents[i, 0], principalComponents[i, 1], principalComponents[i, 2], 
                       color=colors[i % len(colors)], alpha=0.3, s=13)
            ax.text(principalComponents[i, 0], principalComponents[i, 1], principalComponents[i, 2], 
                    str(i+1), color='k', fontsize=2)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('PCA 3D Plot')
        plt.tight_layout()
        plt.savefig(os.path.join(clustering_results_path, 'pca_3d_plot.png'))
        plt.show()








# Function to calculate and plot correlation matrix
def plot_correlation_matrix(principalComponents, features, clustering_results_path):
    principalDf = pd.DataFrame(data=principalComponents, columns=[f'PC{i+1}' for i in range(10)])
    features_df = pd.DataFrame(features, columns=features.columns)

    combined_df = pd.concat([principalDf.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    correlation_matrix = combined_df.corr().loc[principalDf.columns, features_df.columns]
    print(correlation_matrix)

    correlation_table_path = os.path.join(clustering_results_path, 'correlation_matrix_pc_features.csv')
    correlation_matrix.to_csv(correlation_table_path)

    plt.figure(figsize=(20, 15))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('Correlation Matrix between Principal Components and Features', fontsize=12)
    plt.savefig(os.path.join(clustering_results_path, 'correlation_matrix_heatmap.png'))
    plt.show()





# Perform UMAP and plot embeddings
umap_reducer = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7)
standard_embedding = umap_reducer.fit_transform(features_scaled)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = distinct_colors(len(standard_embedding))

for i in range(len(standard_embedding)):
    ax.scatter(standard_embedding[i, 0], standard_embedding[i, 1], standard_embedding[i, 2], 
               color=colors[i % len(colors)], alpha=0.3, s=13)
    ax.text(standard_embedding[i, 0], standard_embedding[i, 1], standard_embedding[i, 2], 
            str(i+1), color='k', fontsize=2)
    
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')
ax.set_title('UMAP Embeddings')
plt.tight_layout()
plt.savefig(os.path.join(clustering_results_path, 'umap_embeddings.png'))
plt.show()






# Function to perform UMAP and plot embeddings to export to the code
def plot_umap_embeddings(features_scaled, clustering_results_path):
    umap_reducer = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7)
    standard_embedding = umap_reducer.fit_transform(features_scaled)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = distinct_colors(len(standard_embedding))

    for i in range(len(standard_embedding)):
        ax.scatter(standard_embedding[i, 0], standard_embedding[i, 1], standard_embedding[i, 2], 
                   color=colors[i % len(colors)], alpha=0.3, s=13)
        ax.text(standard_embedding[i, 0], standard_embedding[i, 1], standard_embedding[i, 2], 
                str(i+1), color='k', fontsize=2)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    ax.set_title('UMAP Embeddings')
    plt.tight_layout()
    plt.savefig(os.path.join(clustering_results_path, 'umap_embeddings.png'))
    plt.show()
    
    return standard_embedding












# Function to calculate and plot distance metrics
def plot_distance_metrics(standard_embedding, clustering_results_path):
    pairwise_dist = pairwise_distances(standard_embedding)
    avg_distance = np.mean(pairwise_dist)
    print(f'Average pairwise distance: {avg_distance}')

    dist_matrix = distance_matrix(standard_embedding, standard_embedding)

    plot = plt.imshow(dist_matrix, cmap='viridis')
    plt.colorbar(plot)
    plt.title('Distance Matrix')
    plt.savefig(os.path.join(clustering_results_path, 'distance_matrix.png'))
    plt.show()

    np.fill_diagonal(dist_matrix, np.nan)
    avg_neighbor_distance = np.nanmean(np.min(dist_matrix, axis=1))
    print(f'Average nearest neighbor distance: {avg_neighbor_distance}')

    # Save connectivity metrics
    with open(os.path.join(clustering_results_path, 'connectivity_metrics.txt'), 'w') as f:
        f.write(f'Average pairwise distance: {avg_distance}\n')
        f.write(f'Average nearest neighbor distance: {avg_neighbor_distance}\n')

    return pairwise_dist, avg_distance, avg_neighbor_distance






# Function to plot density maps
def plot_density_maps(standard_embedding, features, clustering_results_path):
    for feature in features.columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(x=standard_embedding[:, 0], y=standard_embedding[:, 1], hue=features[feature], fill=True, palette='viridis')
        plt.title(f'Density map for {feature}')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.tight_layout()
        plt.savefig(os.path.join(clustering_results_path, f'density_map_{feature}.png'))
        plt.show()





# Function to plot pairwise distance distribution
def plot_pairwise_distance_distribution(pairwise_dist, clustering_results_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(pairwise_dist.flatten(), bins=50, kde=True)
    plt.title('Distribution of Pairwise Distances in UMAP Space')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(clustering_results_path, 'pairwise_distances_distribution.png'))
    plt.show()








# Function for SHAP feature importance analysis
def shap_feature_importance_analysis(umap_reducer, features_scaled, features, clustering_results_path):
    explainer = shap.KernelExplainer(umap_reducer.transform, features_scaled)
    shap_values = explainer.shap_values(features_scaled[:100])

    shap.summary_plot(shap_values, features_scaled[:100], feature_names=features.columns)
    plt.savefig(os.path.join(clustering_results_path, 'shap_summary_plot.png'))
    plt.show()

# # Feature importance analysis using SHAP
explainer = shap.KernelExplainer(umap_reducer.transform, features_scaled)
shap_values = explainer.shap_values(features_scaled[:100])  # Calculate SHAP values for a sample of the data

shap.summary_plot(shap_values, features_scaled[:100], feature_names=features.columns)
plt.savefig(os.path.join(clustering_results_path, 'shap_summary_plot.png'))
plt.show()


# Main execution
principalComponents = perform_pca_analysis(features_scaled, clustering_results_path)
plot_3d_pca(principalComponents, clustering_results_path)
plot_correlation_matrix(principalComponents, features, clustering_results_path)
standard_embedding = plot_umap_embeddings(features_scaled, clustering_results_path)
pairwise_dist, avg_distance, avg_neighbor_distance = plot_distance_metrics(standard_embedding, clustering_results_path)
plot_density_maps(standard_embedding, features, clustering_results_path)
plot_pairwise_distance_distribution(pairwise_dist, clustering_results_path)
shap_feature_importance_analysis( umap_reducer, features_scaled, features, clustering_results_path)
