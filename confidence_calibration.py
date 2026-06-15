import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TemperatureScaling(nn.Module):
    def __init__(self, model):
        super(TemperatureScaling, self).__init__()
        self.model = model
        # Initialize temperature to 1.5
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)
        
    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

def calibrate_model(model, val_loader, device):
    print("Calibrating the model using Temperature Scaling...")
    
    scaled_model = TemperatureScaling(model).to(device)
    
    # Calculate NLL before scaling
    nll_criterion = nn.CrossEntropyLoss()
    scaled_model.eval()
    
    before_nll = 0.0
    for input, label in val_loader:
        input, label = input.to(device), label.to(device)
        with torch.no_grad():
            logits = model(input)
            loss = nll_criterion(logits, label)
            before_nll += loss.item()
            
    print(f"Before Calibration NLL: {before_nll / len(val_loader):.4f}")
    
    # We constrain temperature so it's > 0
    optimizer = optim.LBFGS([scaled_model.temperature], lr=0.01, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        loss = 0.0
        for input, label in val_loader:
            input, label = input.to(device), label.to(device)
            # Forward pass returning calibrated logits
            logits = scaled_model(input)
            loss += nll_criterion(logits, label)
        loss.backward()
        return loss
        
    optimizer.step(eval)
    
    # Calculate NLL after scaling
    after_nll = 0.0
    for input, label in val_loader:
        input, label = input.to(device), label.to(device)
        with torch.no_grad():
            logits = scaled_model(input)
            loss = nll_criterion(logits, label)
            after_nll += loss.item()
            
    print(f"Configured Optimal Temperature: {scaled_model.temperature.item():.4f}")
    print(f"After Calibration NLL: {after_nll / len(val_loader):.4f}")
    
    return scaled_model

def get_probabilities(calibrated_model, X_eval, device):
    calibrated_model.eval()
    tensor_x = torch.Tensor(X_eval.values).to(device)
    with torch.no_grad():
        calibrated_logits = calibrated_model(tensor_x)
        # Apply softmax to get final calibrated probabilities/confidence
        probabilities = F.softmax(calibrated_logits, dim=1)
    return probabilities.cpu().numpy()
