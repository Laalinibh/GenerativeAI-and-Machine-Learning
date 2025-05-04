# This script should be run using torch.distributed.launch or torchrun
# Example: torchrun --nproc_per_node=2 ddp_training.py

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, input_dim=128):
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = SimpleTransformer(input_dim=128, num_classes=10).cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataset = DummyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)

    for epoch in range(5):
        for batch in dataloader:
            x, y = batch
            x, y = x.cuda(rank), y.cuda(rank)
            outputs = ddp_model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch} | Loss: {loss.item()}")

    dist.destroy_process_group()

# Use this function in a script called via torchrun
# Example command: torchrun --nproc_per_node=2 ddp_training.py
# Inside ddp_training.py:
#   from your_module import train
#   import torch.multiprocessing as mp
#   mp.spawn(train, args=(world_size,), nprocs=world_size)
