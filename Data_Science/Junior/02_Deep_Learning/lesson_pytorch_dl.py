"""
================================================================
DS JUNIOR - DEEP LEARNING: PYTORCH FUNDAMENTALS
================================================================

Cài đặt: pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Neural Network Basics:
   - Layers: Linear, Conv2d, LSTM
   - Activations: ReLU, Sigmoid, Softmax
   - Loss: CrossEntropy, MSE
   - Optimizer: SGD, Adam

2. CNN (Convolutional Neural Network):
   - Good for image data
   - Conv2d → BatchNorm → ReLU → Pool
   - Feature extraction

3. RNN/LSTM:
   - Sequential data (text, time series)
   - Hidden state carries memory
   - LSTM solves vanishing gradient

4. Transfer Learning:
   - Use pretrained models
   - Fine-tune for your task
   - Much less data needed
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== BASIC NEURAL NETWORK ==========

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# ========== CNN FOR IMAGES ==========

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ========== LSTM FOR SEQUENCES ==========

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # LSTM returns output and (hidden, cell)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate final hidden states (forward + backward)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        return self.fc(hidden)

# ========== TRAINING LOOP ==========

def train_model(model, train_loader, val_loader, epochs=10, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
        
        scheduler.step()
    
    return model

# ========== TRANSFER LEARNING ==========

def transfer_learning_example():
    """Fine-tune pretrained ResNet"""
    from torchvision import models
    
    # Load pretrained model
    model = models.resnet18(pretrained=True)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Only train new layer
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Train CNN on CIFAR-10:
       - Implement data augmentation
       - Add learning rate warmup
       - Achieve > 80% accuracy

BÀI 2: Sentiment classification với LSTM:
       - Use IMDB dataset
       - Implement attention mechanism
       - Compare with simple averaging

BÀI 3: Transfer learning:
       - Fine-tune ResNet on custom dataset
       - Compare training from scratch vs transfer

BÀI 4: Implement GradCAM:
       - Visualize what CNN "sees"
       - Highlight important regions
"""

if __name__ == "__main__":
    print("=== Deep Learning Demo ===\n")
    
    # Test SimpleNN
    model = SimpleNN(input_size=784, hidden_size=256, num_classes=10)
    x = torch.randn(32, 784)
    out = model(x)
    print(f"SimpleNN: {x.shape} → {out.shape}")
    
    # Test CNN
    cnn = CNN(num_classes=10)
    img = torch.randn(4, 3, 32, 32)
    out = cnn(img)
    print(f"CNN: {img.shape} → {out.shape}")
    
    # Test LSTM
    lstm = LSTMClassifier(vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=2)
    seq = torch.randint(0, 10000, (4, 100))
    out = lstm(seq)
    print(f"LSTM: {seq.shape} → {out.shape}")
    
    print(f"\nTotal CNN params: {sum(p.numel() for p in cnn.parameters()):,}")
