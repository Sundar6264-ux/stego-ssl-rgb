#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

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
        # freeze encoder weights
        for p in self.encoder.parameters():
            p.requires_grad = False
        # attach linear classification head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.encoder(x)
        return self.fc(feats)

def train_probe(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(dataloader, desc="Train Probe", leave=False):
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

def eval_probe(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Eval Probe", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    epochs = 20
    lr = 1e-3

    # data transforms (no stego for probe)
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # data loaders
    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # load pretrained encoder
    encoder = BarlowTwinsEncoder()
    checkpoint = torch.load("bt_bitplane_stego_650ep.pth", map_location=device)
    backbone_state = {
        k.replace('encoder.', ''): v
        for k, v in checkpoint.items() if k.startswith('encoder.')
    }
    encoder.encoder.load_state_dict(backbone_state)
    encoder.to(device)

    # setup linear probe
    model = LinearProbe(encoder, embed_dim=encoder.out_dim, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=0)

    # storage for metrics
    train_losses, train_accs, test_accs = [], [], []

    # training loop
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_probe(model, train_loader, criterion, optimizer, device)
        test_acc = eval_probe(model, test_loader, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc * 100)
        test_accs.append(test_acc * 100)

        print(f"Epoch {epoch:2d} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | "
              f"Test Acc: {test_acc*100:.2f}%")

    # save linear head
    torch.save(model.fc.state_dict(), "linear_probe_fc.pth")

    # ─── PLOT & SAVE FIGURES ─────────────────────────────────────
    epochs_range = np.arange(1, epochs+1)

    # 1) Training Loss
    plt.figure()
    plt.plot(epochs_range, train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss over Epochs")
    plt.tight_layout()
    plt.savefig("linearprobe_train_loss.png")
    plt.close()

    # 2) Train vs Test Accuracy
    plt.figure()
    plt.plot(epochs_range, train_accs, label="Train Acc")
    plt.plot(epochs_range, test_accs,  label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train vs Test Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("linearprobe_accuracy.png")
    plt.close()

    # 3) t-SNE of Test Embeddings (first 2000 samples)
    encoder.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            feats = encoder(x).cpu().numpy()
            all_feats.append(feats)
            all_labels.append(y.cpu().numpy())
    all_feats  = np.vstack(all_feats)[:2000]
    all_labels = np.hstack(all_labels)[:2000]

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    feats_2d = tsne.fit_transform(all_feats)

    plt.figure(figsize=(8,8))
    for cls in range(10):
        idxs = all_labels == cls
        plt.scatter(feats_2d[idxs,0], feats_2d[idxs,1], label=str(cls), s=5)
    plt.legend(title="Class")
    plt.title("t-SNE of Linear Probe Test Embeddings")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig("linearprobe_tsne.png")
    plt.close()

if __name__ == "__main__":
    main()

