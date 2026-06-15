import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class IntrusionDNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[128, 64, 32], dropout_rate=0.3):
        super(IntrusionDNN, self).__init__()
        
        layers = []
        in_features = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
            
        layers.append(nn.Linear(in_features, output_dim))
        # Note: No softmax here, outputting raw logits for calibration
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_model(X_train, y_train, input_dim, output_dim, epochs=30, batch_size=256, lr=0.001):
    print(f"Training DNN on {len(X_train)} samples...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = IntrusionDNN(input_dim=input_dim, output_dim=output_dim).to(device)
    
    # Prepare DataLoader
    tensor_x = torch.Tensor(X_train.values).to(device)
    tensor_y = torch.LongTensor(y_train.values).to(device)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # L2 regularization is added via weight_decay in Adam
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")
            
    print("Training finished.")
    return model, device

def evaluate_logits(model, X_eval, device):
    model.eval()
    tensor_x = torch.Tensor(X_eval.values).to(device)
    with torch.no_grad():
        logits = model(tensor_x)
    return logits.cpu()
