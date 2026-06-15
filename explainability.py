import shap
import torch
import numpy as np

def generate_explanations(model, X_train, X_test, feature_names, device, sample_size=100):
    print(f"Generating SHAP explanations using {sample_size} samples...")
    
    model.eval()
    
    # We use a subset of training data as the background for DeepExplainer
    # Use torch tensor for the explainer
    X_train_tensor = torch.Tensor(X_train.values[:sample_size]).to(device)
    X_test_tensor = torch.Tensor(X_test.values[:sample_size]).to(device)
    
    explainer = shap.DeepExplainer(model, X_train_tensor)
    
    # Calculate SHAP values for test samples
    shap_values = explainer.shap_values(X_test_tensor)
    
    # Sum up SHAP values to explain global feature importance
    print("Saving SHAP summary plot...")
    
    # Handle SHAP output correctly based on whether it returns a list or an array directly
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        
        # Format tensors to standard numpy arrays to prevent dimensionality issues
        if hasattr(shap_values, 'cpu'):
            shap_values = shap_values.cpu().numpy()
            
        if isinstance(shap_values, list):
            sv_list = [sv.cpu().numpy() if hasattr(sv, 'cpu') else sv for sv in shap_values]
            shap.summary_plot(sv_list, X_test.values[:sample_size], feature_names=feature_names, show=False)
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # DeepExplainer returns (samples, classes, features) or (samples, features, classes)
            if shap_values.shape[2] == len(feature_names):
                sv_list = [shap_values[:, i, :] for i in range(shap_values.shape[1])]
            else:
                sv_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
            shap.summary_plot(sv_list, X_test.values[:sample_size], feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X_test.values[:sample_size], feature_names=feature_names, show=False)
            
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        plt.close()
        print("SHAP summary plot saved to shap_summary.png")
    except Exception as e:
        print(f"Failed to generate SHAP plot: {str(e)}")
