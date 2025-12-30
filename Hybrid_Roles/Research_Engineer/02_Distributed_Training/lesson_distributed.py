"""
================================================================
RESEARCH ENGINEER - MODULE 2: DISTRIBUTED TRAINING
================================================================

Training models trên nhiều GPUs/machines
Essential cho large models (LLMs, Vision models)

Cài đặt: pip install torch deepspeed horovod
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Data Parallelism:
   - Copy model to all GPUs
   - Split data across GPUs
   - Average gradients
   - Simple, works for most cases

2. Model Parallelism:
   - Split model across GPUs
   - Needed when model > GPU memory
   - More complex to implement

3. Tensor Parallelism:
   - Split individual layers across GPUs
   - Used in large transformers

4. Pipeline Parallelism:
   - Split model by layers
   - Different layers on different GPUs
   - Micro-batching for efficiency

5. Tools:
   - PyTorch DDP: Native, simple
   - DeepSpeed: Microsoft, ZeRO optimization
   - FSDP: Fully Sharded Data Parallel (PyTorch)
   - Horovod: Uber, MPI-based
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== BASIC DDP SETUP ==========

def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # GPU, use 'gloo' for CPU
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Cleanup distributed training"""
    dist.destroy_process_group()

# ========== DDP TRAINING ==========

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_ddp(rank, world_size, epochs=10):
    """Training function for each process"""
    
    # Setup
    setup_ddp(rank, world_size)
    device = rank
    
    # Model
    model = SimpleCNN().to(device)
    model = DDP(model, device_ids=[rank])
    
    # Data (với DistributedSampler)
    # dataset = YourDataset()
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    # loader = DataLoader(dataset, sampler=sampler, batch_size=32)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        # sampler.set_epoch(epoch)  # Important for shuffling
        
        # Simulate training
        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
        
        # Sync all processes
        dist.barrier()
    
    cleanup_ddp()

# ========== DEEPSPEED CONFIG ==========

DEEPSPEED_CONFIG = """
{
    "train_batch_size": 128,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 1e-6,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 1000
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        },
        "contiguous_gradients": true,
        "overlap_comm": true
    },
    "gradient_clipping": 1.0
}
"""

# DeepSpeed usage pseudo-code:
DEEPSPEED_USAGE = """
import deepspeed

# Initialize
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=deepspeed_config
)

# Training
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()

# Launch script:
# deepspeed train.py --deepspeed_config ds_config.json
"""

# ========== GRADIENT CHECKPOINTING ==========

def train_with_gradient_checkpointing():
    """
    Gradient checkpointing: Trade compute for memory
    Recompute activations during backward instead of storing
    """
    from torch.utils.checkpoint import checkpoint
    
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                ) for _ in range(20)
            ])
        
        def forward(self, x):
            for block in self.blocks:
                # Use checkpointing
                x = checkpoint(block, x, use_reentrant=False)
            return x

# ========== MULTI-NODE TRAINING ==========

SLURM_SCRIPT = """
#!/bin/bash
#SBATCH --job-name=distributed_training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

# Load modules
module load cuda/11.8
module load pytorch/2.0

# Set environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# Launch training
srun python train.py
"""

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Setup DDP training:
       - Train simple model on 2+ GPUs
       - Verify gradient synchronization
       - Compare speed vs single GPU

BÀI 2: Implement DeepSpeed ZeRO Stage 2:
       - Partition optimizer states
       - Measure memory savings

BÀI 3: Gradient Checkpointing:
       - Add to your model
       - Compare memory usage
       - Measure training speed impact

BÀI 4: Model Parallelism:
       - Split large model across 2 GPUs
       - Handle forward/backward properly
       - Profile memory usage
"""

# --- TEST ---
if __name__ == "__main__":
    print("=== Distributed Training Demo ===\n")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # For actual distributed training, use:
    # torch.multiprocessing.spawn(train_ddp, args=(world_size,), nprocs=world_size)
    
    print("\n=== DeepSpeed Config ===")
    print(DEEPSPEED_CONFIG)
