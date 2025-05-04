import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, input_dim=128):
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Mixed Precision
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Scaled gradients to avoid overflow
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"Loss: {loss.item()}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a simple model
    model = SimpleModel(input_dim=128, hidden_dim=256, output_dim=10).to(device)

    # Mixed precision training setup
    scaler = GradScaler()

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Create dataset and dataloader
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the model with gradient checkpointing (For simplicity, we'll omit full checkpointing here)
    for epoch in range(5):
        print(f"Epoch {epoch + 1}")
        train(model, dataloader, criterion, optimizer, scaler, device)

if __name__ == "__main__":
    main()
