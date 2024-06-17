import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
import umap.umap_ as umap
from scipy.stats import ttest_ind, chi2_contingency


# Path to the directory containing the CSV files
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_'

# Path to the metadata CSV file
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'

#save the results
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_sex_clustering_'
if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)
# Get a list of all CSV files in the directory
list_files = glob.glob(os.path.join(features_path, '*.csv'))

# Read all CSV files and concatenate them into a single DataFrame
all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)

# Read metadata
metadata = pd.read_csv(metadata_path)

# Remove rows with null values
all_data = all_data.dropna()

# Add 'Sex' to the dataset by matching 'recording' in all_data with 'Filename' in metadata
all_data = all_data.merge(metadata[['Filename', 'Sex']], left_on='recording', right_on='Filename', how='left')

# Encode 'Sex' using LabelEncoder
label_encoder = LabelEncoder()
all_data['Sex'] = label_encoder.fit_transform(all_data['Sex'])

print(all_data.head(2))

# Create separate DataFrames for male and female
male_data = all_data[all_data['Sex'] == 1]
female_data = all_data[all_data['Sex'] == 0]

# Scale the data for the entire dataset
features = all_data.drop(['Filename', 'Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id', 'Sex'], axis=1)
scaler_all = StandardScaler()
features_scaled = scaler_all.fit_transform(features)

# Scale the data for male dataset
features_male = male_data.drop(['Filename', 'Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id', 'Sex'], axis=1)
scaler_male = StandardScaler()
features_male_scaled = scaler_male.fit_transform(features_male)

# Scale the data for female dataset
features_female = female_data.drop(['Filename', 'Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id', 'Sex'], axis=1)
scaler_female = StandardScaler()
features_female_scaled = scaler_female.fit_transform(features_female)


# UMAP Embedding for male data
umap_reducer_male = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7)
umap_embedding_male = umap_reducer_male.fit_transform(features_male_scaled)

# UMAP Embedding for female data
umap_reducer_female = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7)
umap_embedding_female = umap_reducer_female.fit_transform(features_female_scaled)

# Print the shapes of the scaled features to verify
print(f'All data scaled shape: {features_scaled.shape}')
print(f'Male data scaled shape: {features_male_scaled.shape}')
print(f'Female data scaled shape: {features_female_scaled.shape}')

# You can now proceed with your analysis (e.g., clustering) on the entire dataset, male dataset, and female dataset.

# Plot UMAP embeddings

# Ensure 'colors' is defined correctly before using it for plotting
colors = ['steelblue' if sex == 1 else 'palevioletred' for sex in all_data['Sex']]

# # Plot UMAP embeddings in 3D for male and female separately
# fig = plt.figure(figsize=(20, 10))

# # UMAP embedding for male data
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter(umap_embedding_male[:, 0], umap_embedding_male[:, 1], umap_embedding_male[:, 2], c=[colors[idx] for idx in male_data.index], alpha=0.3, s=10)
# ax1.set_title('UMAP Embedding for Male Calls')
# ax1.set_xlabel('UMAP 1')
# ax1.set_ylabel('UMAP 2')
# ax1.set_zlabel('UMAP 3')

# # UMAP embedding for female data
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.scatter(umap_embedding_female[:, 0], umap_embedding_female[:, 1], umap_embedding_female[:, 2], c=[colors[idx] for idx in female_data.index], alpha=0.3, s=10)
# ax2.set_title('UMAP Embedding for Female Calls')
# ax2.set_xlabel('UMAP 1')
# ax2.set_ylabel('UMAP 2')
# ax2.set_zlabel('UMAP 3')

# plt.tight_layout()
# plt.show()



# plt.figure(figsize=(20, 7))

# plt.subplot(1, 2, 1)
# sns.scatterplot(x=umap_embedding_male[:, 0], y=umap_embedding_male[:, 1], color='lightblue', alpha=0.8)
# plt.title('UMAP Embedding for Male Calls')
# plt.xlabel('UMAP 1')
# plt.ylabel('UMAP 2')

# plt.subplot(1, 2, 2)
# sns.scatterplot(x=umap_embedding_female[:, 0], y=umap_embedding_female[:, 1], color='lightcoral', alpha=0.8)
# plt.title('UMAP Embedding for Female Calls')
# plt.xlabel('UMAP 1')
# plt.ylabel('UMAP 2')

# plt.tight_layout()
# plt.show()
# plt.savefig('C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_sex_clustering_\\umap_embeddings_female_male.png')

colors = ['steelblue' if sex == 1 else 'palevioletred' for sex in all_data['Sex']]

# # UMAP Embedding for all data
# umap_reducer = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7)
# umap_embedding = umap_reducer.fit_transform(features_scaled)

# plt.figure(figsize=(14, 7))
# sns.scatterplot(x=umap_embedding[:, 0], y=umap_embedding[:, 1], hue=all_data['Sex'], palette=('palevioletred','steelblue'), alpha=0.7)
# plt.title('UMAP Embeddings by Sex')
# plt.savefig('C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering\\umap_embeddings_by_sex.png')  
# plt.show()

# # PCA for Feature Importance
# pca = PCA(n_components=2)
# pca_features = pca.fit_transform(features_scaled)

# plt.figure(figsize=(14, 7))
# sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=all_data['Sex'], palette=('palevioletred','steelblue'), alpha=0.7)
# plt.title('PCA Embeddings by Sex')
# plt.savefig('C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering\\pca_embeddings_by_sex.png')  
# plt.show()

# # # Clustering
# n_clusters = 3
# kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features_scaled)
# # agg_clust = AgglomerativeClustering(n_clusters=n_clusters).fit(features_scaled)
# # dbscan = DBSCAN().fit(features_scaled)

# # Evaluate Clustering
# kmeans_silhouette = silhouette_score(features_scaled, kmeans.labels_)
# agg_clust_silhouette = silhouette_score(features_scaled, agg_clust.labels_)
# print(f'KMeans Silhouette Score: {kmeans_silhouette}')
# print(f'Agglomerative Clustering Silhouette Score: {agg_clust_silhouette}')

# # Feature Importance using Random Forest
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(features_scaled, all_data['Sex'])
# feature_importances = pd.DataFrame(rf.feature_importances_, index=features.columns, columns=['importance']).sort_values('importance', ascending=False)
# # save the results to a CSV file
# feature_importances.to_csv(os.path.join(clusterings_results_path, 'feature_importances.csv'))
# # Visualize Feature Importances
# plt.figure(figsize=(12, 8))
# sns.barplot(x=feature_importances.importance, y=feature_importances.index, palette='viridis')
# plt.title('Feature Importances')
# plt.show()
# plt.savefig('C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_sex_clustering_\\feature_importances.png')

# # T-tests for feature differences
# for feature in features.columns:
#     male_values = all_data[all_data['Sex'] == 1][feature]
#     female_values = all_data[all_data['Sex'] == 0][feature]
#     t_stat, p_val = ttest_ind(male_values, female_values)
#     print(f'{feature}: t-statistic={t_stat:.2f}, p-value={p_val:.4f}')
#     save_results = pd.DataFrame({'feature': feature, 't-statistic': t_stat, 'p-value': p_val}, index=[0])
#     save_results.to_csv(os.path.join(clusterings_results_path, f'{feature}_ttest_results.csv'), index=False)

# # Chi-square tests for cluster composition
# kmeans_cluster_composition = pd.crosstab(kmeans.labels_, all_data['Sex'])
# chi2, p, dof, expected = chi2_contingency(kmeans_cluster_composition)
# print(f'Chi-square test (KMeans): chi2={chi2:.2f}, p-value={p:.4f}')
# save_results = pd.DataFrame({'chi2': chi2, 'p-value': p}, index=[0])
# save_results.to_csv(os.path.join(clusterings_results_path, 'kmeans_chi2_results.csv'), index=False)


# Summary Statistics
summary_stats = all_data.groupby('Sex').describe()
print(summary_stats)

# Distribution Analysis - All Features in One Page
features = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id', 'Sex', 'Filename'], axis=1)


# Split features into three parts: two parts of 10 and one part of 6
features_list = features.columns.tolist()
parts = [features_list[:10], features_list[10:20], features_list[20:]]

# Iterate through each part and create a separate subplot for each
for idx, part in enumerate(parts):
    # Define the number of rows and columns for subplots based on part size
    nrows = len(part) // 4 + (len(part) % 4 > 0)
    ncols = 4

    # Create subplots with defined size
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, nrows * 5))
    axes = axes.flatten()

    # Iterate through each feature in the part and plot its distribution by Sex
    for i, feature in enumerate(part):
        sns.histplot(data=all_data, x=feature, hue='Sex', kde=True, stat="density", common_norm=False, palette=('palevioletred','steelblue'), ax=axes[i], alpha=0.7)
        axes[i].set_title(f'Distribution of {feature} by Sex', fontsize=9)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=6)
        axes[i].tick_params(axis='y', labelsize=6)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)  # Adjust spacing between subplots
    plt.show()
    plt.savefig(f'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering\\distribution_analysis_part_{idx + 1}.png')
# # Cluster Composition  
# cluster_composition = pd.crosstab(kmeans.labels_, all_data['Sex'])
# print(cluster_composition)
# save_results = pd.DataFrame(cluster_composition)
# save_results.to_csv(os.path.join(clusterings_results_path, 'cluster_composition.csv'), index=False)