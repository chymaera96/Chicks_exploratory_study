import os
import glob
import pandas as pd
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
from kneed import KneeLocator
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from gap_statistic import OptimalK
import torch
import torch.nn as nn
import torch.optim as optim

# Define paths
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset'
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_vae_clustering_'

if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)

# Load and preprocess data
list_files = glob.glob(os.path.join(features_path, '*.csv'))
all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)
all_data = all_data.dropna()

scaler = StandardScaler()
features = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)

# Define VAE model
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3_mean = nn.Linear(32, latent_dim)
        self.fc3_logvar = nn.Linear(32, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        z_mean = self.fc3_mean(x)
        z_logvar = self.fc3_logvar(x)
        return z_mean, z_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        reconstruction = self.fc3(z)
        return reconstruction

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_logvar

def loss_function(recon_x, x, z_mean, z_logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return recon_loss + kl_div

# Parameters
input_dim = features_scaled.shape[1]
latent_dim = 2
vae = VAE(input_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=0.001)
epochs = 50
batch_size = 32

# Prepare data for PyTorch
dataset = torch.utils.data.TensorDataset(torch.tensor(features_scaled, dtype=torch.float32))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train VAE
for epoch in range(epochs):
    vae.train()
    train_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        recon_x, z_mean, z_logvar = vae(data[0])
        loss = loss_function(recon_x, data[0], z_mean, z_logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {train_loss/len(dataloader.dataset)}')

# Get latent space representation
vae.eval()
with torch.no_grad():
    _, z_mean, _ = vae(torch.tensor(features_scaled, dtype=torch.float32))
    z_mean = z_mean.numpy()

# Perform clustering on latent space
n_max_clusters = 11

kmeans_cluster_evaluation_per_number_clusters = {
    n_clusters: {'silhouette_score': 0,
                 'calinski_harabasz_score': 0,
                 'wcss': 0
                 } for n_clusters in range(2, n_max_clusters)
}

for n_clusters in range(2, n_max_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_membership = kmeans.fit_predict(z_mean)
    
    silhouette = silhouette_score(z_mean, cluster_membership)
    calinski_harabasz = calinski_harabasz_score(z_mean, cluster_membership)
    wcss = kmeans.inertia_
    
    kmeans_cluster_evaluation_per_number_clusters[n_clusters]['silhouette_score'] = silhouette
    kmeans_cluster_evaluation_per_number_clusters[n_clusters]['calinski_harabasz_score'] = calinski_harabasz
    kmeans_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] = wcss
    
    print(f'Number of clusters: {n_clusters}, Silhouette Score: {silhouette}, Calinski Harabasz Score: {calinski_harabasz}, WCSS: {wcss}')

# Save the results
kmeans_cluster_evaluation_per_number_clusters_df = pd.DataFrame(kmeans_cluster_evaluation_per_number_clusters).T
kmeans_cluster_evaluation_per_number_clusters_df.to_csv(os.path.join(clusterings_results_path, 'kmeans_cluster_evaluation_per_number_clusters.csv'))
kmeans_cluster_evaluation_per_number_clusters_df.to_latex(os.path.join(clusterings_results_path, 'kmeans_cluster_evaluation_per_number_clusters.tex'))

# Determine the optimal number of clusters
wcss_vector_across_n_clusters = [kmeans_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] for n_clusters in range(2, n_max_clusters)]
wcss_elbow = KneeLocator(range(2, n_max_clusters), wcss_vector_across_n_clusters, curve='convex', direction='decreasing', interp_method='polynomial', online=False)
best_n_clusters_by_wcss_elbow = wcss_elbow.elbow

optimal_k = OptimalK(parallel_backend='joblib')
n_clusters_optimal_k = optimal_k(features_scaled, cluster_array=np.arange(2, n_max_clusters))
print('Optimal number of clusters according to OptimalK:', n_clusters_optimal_k)
print('Best number of clusters according to the elbow rule with WCSS:', best_n_clusters_by_wcss_elbow)

# Plot evaluation metrics
plt.figure(figsize=(30, 5))
plt.subplot(1, 2, 1)
plt.plot(list(kmeans_cluster_evaluation_per_number_clusters.keys()), [kmeans_cluster_evaluation_per_number_clusters[n_clusters]['silhouette_score'] for n_clusters in kmeans_cluster_evaluation_per_number_clusters.keys()], 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette score per number of clusters')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(list(kmeans_cluster_evaluation_per_number_clusters.keys()), [kmeans_cluster_evaluation_per_number_clusters[n_clusters]['calinski_harabasz_score'] for n_clusters in kmeans_cluster_evaluation_per_number_clusters.keys()], 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Calinski Harabasz Score')
plt.title('Calinski Harabasz Score per number of clusters')
plt.grid()

plt.savefig(os.path.join(clusterings_results_path, 'kmeans_cluster_evaluation_per_number_clusters_1.png'))

# Plot WCSS
plt.figure(figsize=(10, 5))
plt.plot(list(kmeans_cluster_evaluation_per_number_clusters.keys()), [kmeans_cluster_evaluation_per_number_clusters[n_clusters]['wcss'] for n_clusters in kmeans_cluster_evaluation_per_number_clusters.keys()], 'bx-')
plt.axvline(x=best_n_clusters_by_wcss_elbow, color='r', linestyle='--', label=f'Elbow point: {best_n_clusters_by_wcss_elbow}')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('WCSS per number of clusters')
plt.legend()

plt.savefig(os.path.join(clusterings_results_path, 'kmeans_cluster_evaluation_per_number_clusters_2.png'))

plt.show()

print('Clustering with KMeans completed!')

# Visualize latent space
plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=cluster_membership, cmap='viridis')
plt.colorbar(scatter)
plt.title('Latent space clustering')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.savefig(os.path.join(clusterings_results_path, 'latent_space_clustering.png'))
plt.show()
