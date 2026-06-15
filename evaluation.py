import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def evaluate_metrics(y_true, y_pred, y_prob):
    print("Evaluating Model Metrics...")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate ROC-AUC
    # y_prob should be shape (n_samples, n_classes) for ovr
    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except Exception as e:
        print("ROC AUC calculation failed, possibly due to missing classes in test set. Setting to None.")
        roc_auc = None
        
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }

def confidence_decision_system(y_prob, y_pred, y_true, threshold=0.8):
    """
    Confidence-Based Decision System.
    Flags predictions as "Suspicious/Uncertain" if top confidence < threshold.
    """
    print(f"\nEvaluating Confidence-Based Decision System (Threshold: {threshold})")
    
    confidences = np.max(y_prob, axis=1)
    
    # accepted predictions
    accepted_mask = confidences >= threshold
    rejected_mask = ~accepted_mask
    
    num_accepted = np.sum(accepted_mask)
    num_rejected = np.sum(rejected_mask)
    
    print(f"Total samples: {len(y_prob)}")
    print(f"Accepted: {num_accepted} ({(num_accepted/len(y_prob))*100:.2f}%)")
    print(f"Flagged as Uncertain: {num_rejected} ({(num_rejected/len(y_prob))*100:.2f}%)")
    
    if num_accepted > 0:
        acc_accepted = accuracy_score(y_true[accepted_mask], y_pred[accepted_mask])
        print(f"Accuracy on Accepted Samples: {acc_accepted:.4f}")
        
    if num_rejected > 0:
        acc_rejected = accuracy_score(y_true[rejected_mask], y_pred[rejected_mask])
        print(f"Accuracy on Uncertain Samples (which were correctly flagged as risky): {acc_rejected:.4f}")

    return accepted_mask, rejected_mask, confidences
