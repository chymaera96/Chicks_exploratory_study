import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def load_data(file_path, metadata_path):
    """Load and preprocess data from CSV files."""
    list_files = glob.glob(os.path.join(file_path, '*.csv'))
    all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
    metadata = pd.read_csv(metadata_path)
    
    all_data = all_data.dropna()
    all_data = all_data.merge(metadata[['Filename', 'Sex']], left_on='recording', right_on='Filename', how='left')
    
    label_encoder = LabelEncoder()
    all_data['Sex'] = label_encoder.fit_transform(all_data['Sex'])
    
    return all_data

def split_data(all_data, n_males_test=2, n_females_test=2):
    """Split data into train and test sets based on filenames."""
    filename_per_sex = all_data.groupby('Sex')['Filename'].unique()
    female_test = np.random.choice(filename_per_sex[0], n_females_test, replace=False)
    male_test = np.random.choice(filename_per_sex[1], n_males_test, replace=False)
    
    test_filenames = np.concatenate([female_test, male_test])
    test_data = all_data[all_data['Filename'].isin(test_filenames)]
    train_data = all_data[~all_data['Filename'].isin(test_filenames)]
    
    return train_data, test_data

def preprocess_features(data):
    """Preprocess features by dropping unnecessary columns."""
    return data.drop(['Filename', 'Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id'], axis=1)

def select_best_features(X_train, y_train, X_test):
    """Select best features using Random Forest."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(X_train_scaled, y_train)

    selector = SelectFromModel(rf, prefit=True)
    X_train_best = selector.transform(X_train_scaled)
    X_test_best = selector.transform(X_test_scaled)

    selected_feature_names = X_train.columns[selector.get_support()]
    X_train_best = pd.DataFrame(X_train_best, columns=selected_feature_names)
    X_test_best = pd.DataFrame(X_test_best, columns=selected_feature_names)

    return X_train_best, X_test_best, selected_feature_names

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Train and evaluate Decision Tree and SVM models."""
    # Decision Tree
    dt_params = {'criterion': 'entropy', 'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 10}
    dt = DecisionTreeClassifier(**dt_params)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    # SVM
    svm_params = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
    svm = SVC(**svm_params, probability=True)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    return dt, svm, y_pred_dt, y_pred_svm

def plot_roc_curves(dt, svm, X_test, y_test):
    """Plot ROC curves for Decision Tree and SVM."""
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)
    y_score_dt = dt.predict_proba(X_test)[:, 1]
    y_score_svm = svm.predict_proba(X_test)[:, 1]

    fpr_dt, tpr_dt, _ = roc_curve(y_test_binarized, y_score_dt)
    roc_auc_dt = roc_auc_score(y_test_binarized, y_score_dt)

    fpr_svm, tpr_svm, _ = roc_curve(y_test_binarized, y_score_svm)
    roc_auc_svm = roc_auc_score(y_test_binarized, y_score_svm)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_dt, tpr_dt, color='darkorange', lw=2, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
    plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label=f'SVM (AUC = {roc_auc_svm:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def main():
    file_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_'
    metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
    clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\_sex_clustering_'

    if not os.path.exists(clusterings_results_path):
        os.makedirs(clusterings_results_path)

    all_data = load_data(file_path, metadata_path)
    train_data, test_data = split_data(all_data)

    features_train = preprocess_features(train_data)
    features_test = preprocess_features(test_data)

    X_train = features_train.drop(['Sex'], axis=1)
    y_train = features_train['Sex']
    X_test = features_test.drop(['Sex'], axis=1)
    y_test = features_test['Sex']

    X_train_best, X_test_best, selected_features = select_best_features(X_train, y_train, X_test)

    print("Selected Features:\n", selected_features)
    print("First two rows of the training set with selected features:\n", X_train_best.head(2))
    print("First two rows of the test set with selected features:\n", X_test_best.head(2))

    dt, svm, y_pred_dt, y_pred_svm = train_and_evaluate_models(X_train_best, y_train, X_test_best, y_test)

    print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

    print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

    plot_roc_curves(dt, svm, X_test_best, y_test)

if __name__ == "__main__":
    main()