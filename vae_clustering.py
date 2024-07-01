import os
import glob
import pandas as pd
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from clustering_utils import get_random_samples, plot_and_save_audio_segments, statistical_report, create_statistical_report_with_radar_plots

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

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
latent_dim = 2

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3_mean = nn.Linear(32, latent_dim)
        self.fc3_log_var = nn.Linear(32, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        z_mean = self.fc3_mean(x)
        z_log_var = self.fc3_log_var(x)
        z = z_mean + torch.exp(0.5 * z_log_var) * torch.randn_like(z_mean)
        return z_mean, z_log_var, z

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        reconstruction = self.fc3(x)
        return reconstruction

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

def loss_function(reconstruction, x, z_mean, z_log_var):
    reconstruction_loss = nn.functional.mse_loss(reconstruction, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return reconstruction_loss + kl_loss

# Prepare data for PyTorch
tensor_features = torch.tensor(features_scaled, dtype=torch.float32)
dataset = TensorDataset(tensor_features)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize and train VAE
input_dim = features_scaled.shape[1]
encoder = Encoder(input_dim, latent_dim)
decoder = Decoder(latent_dim, input_dim)
vae = VAE(encoder, decoder)

optimizer = optim.Adam(vae.parameters(), lr=0.001)
num_epochs = 50

vae.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        reconstruction, z_mean, z_log_var = vae(x)
        loss = loss_function(reconstruction, x, z_mean, z_log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader.dataset)}')

# Get latent space representation
vae.eval()
with torch.no_grad():
    z_means = []
    for batch in dataloader:
        x = batch[0]
        z_mean, _, _ = vae.encoder(x)
        z_means.append(z_mean)
    z = torch.cat(z_means).numpy()

# Perform clustering on latent space
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_membership = kmeans.fit_predict(z)

all_data['cluster_membership'] = cluster_membership
all_data.to_csv(os.path.join(clusterings_results_path, f'vae_clustering_{n_clusters}_membership.csv'), index=False)

# Visualize latent space
plt.figure(figsize=(10, 8))
scatter = plt.scatter(z[:, 0], z[:, 1], c=cluster_membership, cmap='viridis')
plt.colorbar(scatter)
plt.title('VAE Latent Space')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.savefig(os.path.join(clusterings_results_path, 'vae_latent_space.png'))
plt.close()

# Get random samples and plot audio segments
random_samples = get_random_samples(all_data, 'cluster_membership', num_samples=5)
plot_and_save_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')

# Generate statistical reports
stats = statistical_report(all_data, cluster_membership, n_clusters, metadata, clusterings_results_path)
print(stats)

radar = create_statistical_report_with_radar_plots(all_data, cluster_membership, n_clusters, metadata, clusterings_results_path)
