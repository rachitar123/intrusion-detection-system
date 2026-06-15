from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def select_features(X_train, y_train, num_features=25):
    print(f"Running ExtraTreesClassifier to select top {num_features} features...")
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_indices = indices[:num_features]
    top_features = X_train.columns[top_indices].tolist()
    
    print(f"Top {num_features} selected features: {top_features}")
    
    X_train_selected = X_train.iloc[:, top_indices]
    
    return X_train_selected, top_features, importances, indices

def apply_selected_features(X_test, top_features):
    return X_test[top_features]

def plot_feature_importances(importances, indices, feature_names, top_n=20, save_path="feature_importances.png"):
    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances")
    
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
    plt.xlim([-1, top_n])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Feature importances plot saved to {save_path}")
