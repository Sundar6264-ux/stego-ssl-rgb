#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class TwoCropsDataset(Dataset):
    """Dataset wrapper that returns two augmented views of each image."""
    def __init__(self, base_dataset, transform1, transform2):
        self.base_dataset = base_dataset
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        return self.transform1(img), self.transform2(img)


class BitPlaneStego(nn.Module):
    """
    Embeds a fixed pseudo-random bit-plane into the green channel of an image.

    Args:
        bit_plane: which bit to embed (0 = LSB, 1 = next bit, etc.)
        image_size: spatial size of the (square) image
        seed: random seed to generate the key
    """
    def __init__(self, bit_plane: int = 0, image_size: int = 32, seed: int = 42):
        super().__init__()
        self.bit = bit_plane
        torch.manual_seed(seed)
        key = torch.randint(
            0, 2, (1, image_size, image_size), dtype=torch.uint8
        )
        self.register_buffer('key', key)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: FloatTensor [C, H, W] in [0,1]
        x = (img * 255).to(torch.uint8)
        mask = ~(1 << self.bit)
        g = x[1] & mask
        g |= (self.key << self.bit).squeeze(0)
        x[1] = g
        return x.to(torch.float32).div(255.)


class BarlowTwins(nn.Module):
    """
    Barlow Twins model: encoder + projector.
    """
    def __init__(self, backbone: nn.Module, embed_dim: int = 512, projector_dim: int = 2048):
        super().__init__()
        if hasattr(backbone, 'fc'):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            in_features = embed_dim
        self.encoder = backbone
        self.projector = nn.Sequential(
            nn.Linear(in_features, projector_dim, bias=False),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_dim, projector_dim, bias=False),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_dim, projector_dim, bias=False),
            nn.BatchNorm1d(projector_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x)
        z = self.projector(y)
        return z


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambda_coeff: float = 5e-3, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the Barlow Twins loss with epsilon stabilization.
    """
    N, D = z1.size()
    z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + eps)
    z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + eps)
    c = torch.mm(z1_norm.T, z2_norm) / N
    on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
    off_diag = c.flatten().pow(2).sum() - torch.diagonal(c).pow(2).sum()
    return on_diag + lambda_coeff * off_diag


def train_ssl(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
              scheduler: CosineAnnealingLR, device: torch.device, epochs: int):
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (x1, x2) in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{epochs}"), start=1):
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1), model(x2)
            loss = barlow_twins_loss(z1, z2)
            if torch.isnan(loss):
                print(f"⚠️ NaN detected at epoch {epoch}, batch {batch_idx}")
                raise RuntimeError("NaN loss—exiting to debug")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 50 == 0 or batch_idx == len(loader):
                print(f"    Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    torch.save(model.state_dict(), "bt_bitplane_stego_650ep.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    epochs = 650  # extended training
    lr = 1e-3

    base_aug = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ])

    transform1 = transforms.Compose([
        base_aug,
        transforms.ToTensor(),
    ])
    transform2 = transforms.Compose([
        base_aug,
        transforms.ToTensor(),
        BitPlaneStego(bit_plane=1, image_size=32, seed=42)
    ])

    cifar_train = datasets.CIFAR10(root="./data", train=True, download=True)
    train_ds = TwoCropsDataset(cifar_train, transform1, transform2)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    backbone = models.resnet18(weights=None)
    model = BarlowTwins(backbone)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_ssl(model, train_loader, optimizer, scheduler, device, epochs)

