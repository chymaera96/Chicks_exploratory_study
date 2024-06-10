import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import skfuzzy as fuzz
import umap.umap_ as umap
from scipy.stats import ttest_ind, chi2_contingency

# Paths to your data
file_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
clustering_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering\\_individual_clustering_results'

# Read metadata
metadata = pd.read_csv(metadata_path)

# Get all CSV files
list_files = glob.glob(os.path.join(file_path, '*.csv'))

# Concatenate all data into a single DataFrame
all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)

# Remove rows with null values
all_data = all_data.dropna()

# Standardize the features
scaler = StandardScaler()
features = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording','call_id'], axis=1)  # Remove non-numeric columns
features_scaled = scaler.fit_transform(features)

# Add 'Sex' to the dataset
all_data['Sex'] = metadata['Sex']

# Encode the 'recording' variable
label_encoder = LabelEncoder()
recording_encoded = label_encoder.fit_transform(all_data['recording'])

# Concatenate standardized features with encoded 'recording'
features_encoded = np.concatenate([features_scaled, recording_encoded.reshape(-1, 1)], axis=1)


import umap.umap_ as umap

# UMAP Embedding
umap_reducer = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7)
standard_embedding = umap_reducer.fit_transform(features_encoded)

plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=10, alpha=0.5)
plt.title('UMAP with 30 neighbors, 9 components and 0.7 min_dist')
plt.show()

# Initialize and define the colors for the plot as light blue for M and light red for F   
colors = {'F': 'lightcoral', 'M': 'lightblue'}
silhouette_scores = []
wcss = []
calinski_harabasz_scores = []
fpcs = []

# Create subplots for each number of clusters
fig1, axes1 = plt.subplots(3, 3, figsize=(12, 12))

# Iterate over the number of clusters
for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    try:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(features_encoded.transpose(),
                                                         ncenters, 2, error=0.005, maxiter=1000, init=None)
        
        # Transform centroids into the UMAP space using the weighted mean
        umap_centroids = np.dot(u, standard_embedding) / np.sum(u, axis=1)[:, None]
        
        # Determine cluster membership
        cluster_membership = np.argmax(u, axis=0)

        silhouette_scores.append(silhouette_score(features_encoded, cluster_membership))
        calinski_harabasz_scores.append(calinski_harabasz_score(features_encoded, cluster_membership))

        # Calculate and store the WCSS (Elbow Method)
        cntr_transposed = cntr.T
        distances = np.linalg.norm(features_encoded[:, :, np.newaxis] - cntr_transposed, axis=1)
        closest_distances = np.min(distances, axis=1)
        wcss_n = np.sum(closest_distances**2)
        wcss.append(wcss_n)

        fpcs.append(fpc)

        # Plot
        for j in range(ncenters):
            for sex_label, color in colors.items():
                mask = (all_data['Sex'] == sex_label) & (cluster_membership == j)
                ax.plot(standard_embedding[mask, 0],
                        standard_embedding[mask, 1],
                        marker='.', linestyle='', color=color, alpha=0.5)       
        ax.plot(umap_centroids[:, 0], umap_centroids[:, 1], 'rx', markersize=8, markeredgewidth=1)
        ax.set_title(f'Centers = {ncenters}; FPC = {fpc:.2f}')
        ax.axis('off')
    except Exception as e:
        print(f"Error processing {ncenters} centers: {e}")
        silhouette_scores.append(np.nan)
        calinski_harabasz_scores.append(np.nan)
        wcss.append(np.nan)
        fpcs.append(np.nan)

plt.tight_layout()
plt.show()

# Create a DataFrame for the results
Results = pd.DataFrame({
    'N_clusters': range(2, 11),
    'Silhouette_score': silhouette_scores,
    'Calinski_harabasz_score': calinski_harabasz_scores,
    'Within-Cluster Sum of Squares': wcss
})

# Remove any NaN values
Results.dropna(inplace=True)

# Find the best number of clusters
best_silhouette_n_clusters = Results.iloc[Results['Silhouette_score'].idxmax()]['N_clusters']
best_calinski_n_clusters = Results.iloc[Results['Calinski_harabasz_score'].idxmax()]['N_clusters']
best_wcss_n_clusters = Results.iloc[Results['Within-Cluster Sum of Squares'].idxmin()]['N_clusters']

# Print the best number of clusters
print(f"Best number of clusters by Silhouette Score: {best_silhouette_n_clusters}")
print(f"Best number of clusters by Calinski Harabasz Score: {best_calinski_n_clusters}")
print(f"Best number of clusters by WCSS: {best_wcss_n_clusters}")

# Plot results
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(Results['N_clusters'], Results['Silhouette_score'], marker='o', linestyle='-', color='b')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(Results['N_clusters'], Results['Within-Cluster Sum of Squares'], marker='o', linestyle='-', color='r')
plt.title('WCSS (Elbow Method) vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(Results['N_clusters'], Results['Calinski_harabasz_score'], marker='o', linestyle='-', color='g')
plt.title('Calinski-Harabasz Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Score')
plt.grid(True)

plt.tight_layout()
plt.show()


# Summary Statistics
summary_stats = all_data.groupby('Sex').describe()
print(summary_stats)

# Distribution Analysis - All Features in One Page
features = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id', 'Sex'], axis=1)

fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(20, 35))  # Adjusted for 26 features (7x4 grid)
axes = axes.flatten()

for i, feature in enumerate(features.columns):
    sns.histplot(data=all_data, x=feature, hue='Sex', kde=True, stat="density", common_norm=False, 
                 palette={'F': 'lightcoral', 'M': 'lightblue'}, ax=axes[i])
    axes[i].set_title(f'Distribution of {feature} by Sex')

# Hide any unused subplots (if the number of features is less than 28)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Cluster Composition
cluster_composition = pd.crosstab(cluster_membership, all_data['Sex'])
print(cluster_composition)

# t-tests
for feature in features.columns:
    male_values = all_data[all_data['Sex'] == 'M'][feature]
    female_values = all_data[all_data['Sex'] == 'F'][feature]
    t_stat, p_val = ttest_ind(male_values, female_values)
    print(f'{feature}: t-statistic={t_stat:.2f}, p-value={p_val:.4f}')

# Chi-square tests for cluster composition
chi2, p, dof, expected = chi2_contingency(cluster_composition)
print(f'Chi-square test: chi2={chi2:.2f}, p-value={p:.4f}')

# UMAP Plots by Sex
plt.figure(figsize=(14, 7))
sns.scatterplot(x=standard_embedding[:, 0], y=standard_embedding[:, 1], hue=all_data['Sex'], palette=['lightblue', 'lightcoral'], alpha=0.5)
plt.title('UMAP Embeddings by Sex')
plt.show()
