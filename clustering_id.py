import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from prince import MCA  # Importa la classe MCA dalla libreria prince
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from prince import MCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import shap
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.stats import ttest_ind

# Definisci i percorsi dei file
file_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
clustering_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering\\_individual_clustering_results'
# Check if the directory exists, if not, create it
if not os.path.exists(clustering_results_path):
    os.makedirs(clustering_results_path)


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

# Concatena le features standardizzate con recording_encoded
features_encoded = np.column_stack((features_scaled, recording_encoded))


# Converti l'array numpy in un DataFrame di pandas
features_encoded_df = pd.DataFrame(features_encoded)

# # Esegui l'Analisi delle Corrispondenze Multiple (MCA) con le features codificate
# mca = MCA(n_components=2)
# mca_results = mca.fit(features_encoded_df)


# # Salva i risultati dell'MCA in un file CSV
# mca_results_df = pd.DataFrame(mca.row_coordinates(features_encoded_df), columns=['Component 1', 'Component 2'])
# mca_results_df.to_csv(os.path.join(clustering_results_path, 'mca_results.csv'), index=False)



# Analisi Discriminante Lineare (LDA)
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

# Feature Importance usando SHAP
explainer = shap.KernelExplainer(lda.predict, features_scaled)
shap_values = explainer.shap_values(features_scaled)  # SHAP values per un campione di dati

shap.summary_plot(shap_values, features_scaled, feature_names=all_data.columns.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id']))
plt.savefig(os.path.join(clustering_results_path, 'shap_summary_plot.png'))
plt.show()

# # Clustering su Sottogruppi
# from sklearn.cluster import KMeans

# for label in np.unique(recording_encoded):
#     label_data = features_scaled[recording_encoded == label]
#     kmeans = KMeans(n_clusters=3)
#     clusters = kmeans.fit_predict(label_data)

#     plt.figure(figsize=(8, 6))
#     plt.scatter(label_data[:, 0], label_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
#     plt.title(f'Clustering for {label_encoder.inverse_transform([label])[0]}')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.tight_layout()
#     plt.savefig(os.path.join(clustering_results_path, f'clustering_{label_encoder.inverse_transform([label])[0]}.png'))
#     plt.show()

# # Visualizzazione delle Caratteristiche usando t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(features_scaled)

# fig, ax = plt.subplots(figsize=(10, 8))
# for label in np.unique(recording_encoded):
#     ax.scatter(tsne_results[recording_encoded == label, 0],
#                tsne_results[recording_encoded == label, 1],
#                label=label_encoder.inverse_transform([label])[0],
#                alpha=0.6)
# ax.set_title('t-SNE: Projection of Features')
# ax.set_xlabel('t-SNE 1')
# ax.set_ylabel('t-SNE 2')
# ax.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(clustering_results_path, 'tsne_plot.png'))
# plt.show()

# # Esegui il test T e salva i risultati
# results = []

# feature_names = all_data.columns.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording'])
# for feature in feature_names:
#     feature_data = all_data[feature]
#     groups = [feature_data[recording_encoded == label] for label in np.unique(recording_encoded)]
#     if len(groups) >= 2:
#         stat, p_value = ttest_ind(groups[0], groups[1])  # Confronta le prime due classi
#         results.append((feature, stat, p_value))

#         print(f'T-Test per {feature}: Statistica={stat}, p-value={p_value}')
#         with open(os.path.join(clustering_results_path, 'statistical_analysis.txt'), 'a') as f:
#             f.write(f'T-Test per {feature}: Statistica={stat}, p-value={p_value}\n')

# # Salva i risultati in un file CSV
# results_df = pd.DataFrame(results, columns=['Feature', 'Statistic', 'p-value'])
# results_df.to_csv(os.path.join(clustering_results_path, 'statistical_analysis_results.csv'), index=False)

# # Visualizza i risultati con un grafico a barre dei p-value
# plt.figure(figsize=(12, 8))
# plt.bar(results_df['Feature'], results_df['p-value'], color='blue')
# plt.axhline(y=0.05, color='r', linestyle='--')  # Linea di soglia per significatività statistica
# plt.xlabel('Features')
# plt.ylabel('p-value')
# plt.xticks(rotation=90)
# plt.title('p-value per ciascuna caratteristica')
# plt.tight_layout()
# plt.savefig(os.path.join(clustering_results_path, 'p_value_barplot.png'))
# plt.show()