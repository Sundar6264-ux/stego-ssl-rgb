#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ---- Barlow Twins encoder definition (ResNet18 backbone) ----
class BarlowTwinsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.out_dim = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

# ---- Linear probe model ----
class LinearProbe(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():  # freeze encoder
            p.requires_grad = False
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.encoder(x)
        return self.fc(feats)


def train_probe(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def eval_probe(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    epochs = 100

    # Data transforms
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Data loaders
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load pretrained encoder
    encoder = BarlowTwinsEncoder().to(device)
    ckpt = torch.load("bt_bitplane_stego_650ep.pth", map_location=device)
    backbone_state = {k.replace('encoder.', ''): v for k, v in ckpt.items() if k.startswith('encoder.')}
    encoder.encoder.load_state_dict(backbone_state)

    # Linear probe setup
    model = LinearProbe(encoder, embed_dim=encoder.out_dim, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.fc.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Training and evaluation
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_probe(model, train_loader, criterion, optimizer, device)
        test_acc = eval_probe(model, test_loader, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%, LR={scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()

    # Save the linear head
    torch.save(model.fc.state_dict(), "linear_probe_fc_adamw.pth")

if __name__ == "__main__":
    main()

