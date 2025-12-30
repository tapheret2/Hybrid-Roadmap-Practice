"""
================================================================
RESEARCH ENGINEER - MODULE 1: DEEP LEARNING ADVANCED
================================================================

Research Engineer implement cutting-edge ML/DL từ papers
Focus: PyTorch, custom architectures, advanced training techniques

Cài đặt: pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. PyTorch vs TensorFlow:
   - PyTorch: Dynamic graph, pythonic, research-friendly
   - TensorFlow: Static graph (now eager), production-ready
   - JAX: Functional, JIT compilation, Google Research

2. Advanced Architectures:
   - Transformers: Attention is All You Need
   - ViT: Vision Transformer
   - Diffusion Models: DDPM, Stable Diffusion
   - GAN: Generative Adversarial Networks

3. Training Techniques:
   - Mixed Precision (FP16): Faster training, less memory
   - Gradient Accumulation: Simulate larger batch sizes
   - Learning Rate Scheduling: Warmup, Cosine decay
   - Gradient Clipping: Prevent exploding gradients

4. Regularization:
   - Dropout, DropPath
   - Label Smoothing
   - Mixup, CutMix
   - Weight Decay
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== CUSTOM DATASET ==========

class CustomDataset(Dataset):
    """Example custom dataset"""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return torch.FloatTensor(x), torch.LongTensor([y])

# ========== TRANSFORMER BLOCK ==========

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class TransformerBlock(nn.Module):
    """Single Transformer encoder block"""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ========== VISION TRANSFORMER (ViT) ==========

class VisionTransformer(nn.Module):
    """Simplified Vision Transformer"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Classification head (use class token)
        return self.head(x[:, 0])

# ========== ADVANCED TRAINING LOOP ==========

def train_with_advanced_techniques(model, train_loader, val_loader, epochs=10):
    """Training loop with advanced techniques"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=0.01
    )
    
    # Learning rate scheduler (warmup + cosine decay)
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).squeeze()
            
            optimizer.zero_grad()
            
            # Mixed precision forward
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Mixed precision backward
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Implement ViT từ paper gốc:
       - Thêm DropPath (stochastic depth)
       - Thêm nhiều augmentations (RandAugment)
       - Train trên CIFAR-10

BÀI 2: Implement BERT-style pretraining:
       - Masked Language Modeling
       - Next Sentence Prediction

BÀI 3: Build custom loss function:
       - Focal Loss cho imbalanced data
       - Contrastive Loss cho similarity learning

BÀI 4: Implement Diffusion Model basics:
       - Forward process (add noise)
       - Reverse process (denoise)
       - Simple UNet architecture
"""

# --- TEST ---
if __name__ == "__main__":
    print("=== Deep Learning Advanced Demo ===\n")
    
    # Test Transformer block
    block = TransformerBlock(embed_dim=256, num_heads=8)
    x = torch.randn(2, 16, 256)  # (batch, seq_len, embed_dim)
    out = block(x)
    print(f"Transformer Block: {x.shape} → {out.shape}")
    
    # Test ViT
    vit = VisionTransformer(
        img_size=224, patch_size=16, in_channels=3,
        num_classes=10, embed_dim=384, depth=6, num_heads=6
    )
    img = torch.randn(2, 3, 224, 224)
    out = vit(img)
    print(f"ViT: {img.shape} → {out.shape}")
    print(f"Total params: {sum(p.numel() for p in vit.parameters()):,}")
