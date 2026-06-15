import os
import torch
import numpy as np
import warnings
from torch.utils.data import TensorDataset, DataLoader

# Import custom modules
from download_data import prepare_data
from data_preprocessing import load_data, preprocess_features, apply_smote
from feature_selection import select_features, apply_selected_features, plot_feature_importances
from model import train_model, evaluate_logits
from confidence_calibration import calibrate_model, get_probabilities
import evaluation
import visualization
import explainability

warnings.filterwarnings('ignore')

def main():
    print("=== Confidence-Driven Multiclass Network Intrusion Detection ===")
    
    # 1. Provide Data
    prepare_data()
    train_df = load_data("./data/KDDTrain+.txt")
    test_df = load_data("./data/KDDTest+.txt")
    
    # 2. Preprocess Data
    X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, target_encoder = preprocess_features(train_df, test_df)
    
    # Retrieve class names
    classes = target_encoder.classes_
    print(f"Detected {len(classes)} classes: {classes}")
    
    # Handle Imbalance
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train_enc)
    
    # 3. Feature Selection
    num_features = 20
    X_train_selected, top_features, importances, indices = select_features(X_train_resampled, y_train_resampled, num_features)
    X_test_selected = apply_selected_features(X_test_scaled, top_features)
    
    # Plot feature importances
    plot_feature_importances(importances, indices, X_train_resampled.columns.tolist(), top_n=20)
    
    # 4. Model Architecture (Train DNN)
    input_dim = num_features
    output_dim = len(classes)
    
    model, device = train_model(X_train_selected, y_train_resampled, input_dim, output_dim, epochs=15, batch_size=256)
    
    # Evaluate Baseline (Uncalibrated)
    print("\n--- Evaluating Baseline (Uncalibrated) ---")
    uncalibrated_logits = evaluate_logits(model, X_test_selected, device)
    uncalibrated_probs = torch.nn.functional.softmax(uncalibrated_logits, dim=1).numpy()
    uncalibrated_preds = np.argmax(uncalibrated_probs, axis=1)
    
    evaluation.evaluate_metrics(y_test_enc, uncalibrated_preds, uncalibrated_probs)
    
    # 5. Confidence Calibration
    val_size = int(len(X_test_selected) * 0.2)
    X_calib = X_test_selected[:val_size]
    y_calib = y_test_enc[:val_size]
    X_eval = X_test_selected[val_size:]
    y_eval = y_test_enc[val_size:]
    
    calib_tensor_x = torch.Tensor(X_calib.values)
    calib_tensor_y = torch.LongTensor(y_calib.values)
    calib_loader = DataLoader(TensorDataset(calib_tensor_x, calib_tensor_y), batch_size=128, shuffle=False)
    
    calibrated_model = calibrate_model(model, calib_loader, device)
    
    # 6. Evaluation on test subset
    print("\n--- Evaluating Calibrated Model ---")
    calibrated_probs = get_probabilities(calibrated_model, X_eval, device)
    calibrated_preds = np.argmax(calibrated_probs, axis=1)
    
    metrics = evaluation.evaluate_metrics(y_eval, calibrated_preds, calibrated_probs)
    
    # 7. Confidence Decision System
    threshold = 0.8
    _, _, confidences = evaluation.confidence_decision_system(calibrated_probs, calibrated_preds, y_eval.values, threshold)
    
    # 8. Visualization
    visualization.plot_confusion_matrix(metrics['confusion_matrix'], classes)
    visualization.plot_multiclass_roc(y_eval.values, calibrated_probs, classes)
    visualization.plot_confidence_distribution(confidences)
    
    # 9. Explainability
    explainability.generate_explanations(
        model=model, 
        X_train=X_train_selected, 
        X_test=X_eval, 
        feature_names=top_features, 
        device=device,
        sample_size=100
    )

if __name__ == "__main__":
    main()
