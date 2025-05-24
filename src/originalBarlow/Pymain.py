import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import torchvision.models as models
from tqdm import tqdm

# ─── 1) Two‑Crop Transform ───────────────────────────────────────────────────
class TwoCropsTransform:
    """Create two random augmented views of the same image."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

# ─── 2) Barlow Twins Model ───────────────────────────────────────────────────
class BarlowTwinsModel(nn.Module):
    def __init__(self, backbone='resnet50', proj_hidden_dim=512, out_dim=512):
        super().__init__()
        resnet = getattr(models, backbone)(zero_init_residual=True)
        # Encoder backbone up to avgpool
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1, resnet.layer2,
            resnet.layer3, resnet.layer4, resnet.avgpool
        )
        enc_dim = resnet.fc.in_features

        # Projector MLP
        self.projector = nn.Sequential(
            nn.Linear(enc_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)
        )

    def forward(self, x):
        h = self.encoder(x).flatten(start_dim=1)
        return self.projector(h)

# ─── 3) Barlow Twins Loss ─────────────────────────────────────────────────────
def off_diagonal(mat):
    n, _ = mat.shape
    return mat.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()

def barlow_twins_loss(z1, z2, lambd=2e-3, eps=1e-5):
    B, D = z1.size()
    z1n = (z1.float() - z1.float().mean(0, keepdim=True)) / (z1.float().std(0, keepdim=True) + eps)
    z2n = (z2.float() - z2.float().mean(0, keepdim=True)) / (z2.float().std(0, keepdim=True) + eps)
    c = (z1n.T @ z2n) / B
    c = torch.nan_to_num(c, nan=0.0, posinf=1.0, neginf=-1.0)
    on_diag  = (torch.diagonal(c) - 1).pow(2).sum()
    off_diag = off_diagonal(c).pow(2).sum()
    return on_diag + lambd * off_diag

# ─── 4) DataLoader ───────────────────────────────────────────────────────────
def get_ssl_dataloader(batch_size=256, num_workers=4):
    transform = T.Compose([
        T.RandomResizedCrop(32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=10),
        T.RandomApply([T.ColorJitter(0.8,0.8,0.8,0.2)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1,2.0)),
        T.ToTensor(),
        T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    ds = CIFAR10(root='./data', train=True,
                 transform=TwoCropsTransform(transform),
                 download=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)

def get_supervised_loader(batch_size=256, num_workers=4, train=True):
    transform = T.Compose([
        T.RandomCrop(32, padding=4) if train else T.Resize(32),
        T.RandomHorizontalFlip() if train else nn.Identity(),
        T.ToTensor(),
        T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    ds = CIFAR10(root='./data', train=train, transform=transform, download=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, pin_memory=True)

# ─── 5) SSL Training Loop ────────────────────────────────────────────────────
def train_barlow_twins(model, loader, epochs=800, base_lr=0.2, wd=1e-4, device='cuda'):
    model.to(device)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.float()

    scaled_lr = base_lr * (loader.batch_size / 256)
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs-10)
    scaler    = GradScaler("cuda")   # <— positional device

    for epoch in range(1, epochs+1):
        if epoch <= 10:
            lr = scaled_lr * (epoch / 10)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"SSL Epoch {epoch}/{epochs}", ncols=100)
        for (x1, x2), _ in pbar:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):        # <— positional device
                z1 = model(x1)
                z2 = model(x2)
                loss = barlow_twins_loss(z1, z2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * x1.size(0)
            pbar.set_postfix(loss=total_loss/((pbar.n+1)*loader.batch_size))
        scheduler.step()

# ─── 6) Linear Probe ─────────────────────────────────────────────────────────
class LinearProbe(nn.Module):
    def __init__(self, encoder, feat_dim, n_classes=10):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x).flatten(1)
        return self.fc(h)

def train_linear_probe(model, train_loader, val_loader,
                       epochs=100, lr=0.1, mixup_alpha=1.0,
                       label_smoothing=0.1, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(1, epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(y.size(0))
            x = lam * x + (1-lam) * x[idx]
            y_a, y_b = y, y[idx]

            logits = model(x)
            loss = lam*criterion(logits, y_a) + (1-lam)*criterion(logits, y_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Linear probe epoch {epoch}/{epochs} complete.")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds==y).sum().item()
            total   += y.size(0)
    acc = 100. * correct/total
    print(f"🔹 Linear probe accuracy: {acc:.2f}%")
    return acc

# ─── 7) Full Fine‑tune ────────────────────────────────────────────────────────
def finetune(model, train_loader, val_loader,
             epochs=50, lr=1e-3, wd=1e-4, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(1, epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds==y).sum().item()
            total   += y.size(0)
    acc = 100. * correct/total
    print(f"🔹 Fine‑tune accuracy: {acc:.2f}%")
    return acc

# ─── 8) Main Entrypoint ──────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # 1) SSL Pre‑training
    ssl_loader = get_ssl_dataloader(batch_size=256)
    bt_model = BarlowTwinsModel(backbone='resnet50',
                                proj_hidden_dim=512,
                                out_dim=512)
    train_barlow_twins(bt_model, ssl_loader,
                       epochs=800, base_lr=0.2, wd=1e-4,
                       device=device)

    # 2) Save SSL model (no directories)
    torch.save(bt_model.state_dict(), 'bt_res50_512_lambda1e3.pth')
    print("✅ SSL pre‑training complete. Model saved to bt_res50_512_lambda1e3.pth\n")

    # 3) Linear Probe
    feat_dim = bt_model.projector[0].in_features
    probe = LinearProbe(bt_model.encoder, feat_dim, n_classes=10)
    train_loader = get_supervised_loader(batch_size=256, train=True)
    val_loader   = get_supervised_loader(batch_size=256, train=False)
    _ = train_linear_probe(probe, train_loader, val_loader,
                            epochs=100, lr=0.1,
                            mixup_alpha=1.0,
                            label_smoothing=0.1,
                            device=device)

    # 4) Full Fine‑tune (optional)
    head = nn.Linear(feat_dim, 10).to(device)
    full_model = nn.Sequential(
        bt_model.encoder,
        nn.Flatten(),
        head
    )
    _ = finetune(full_model, train_loader, val_loader,
                 epochs=50, lr=1e-3, wd=1e-4,
                 device=device)

    # 5) Save fine‑tuned model
    torch.save(full_model.state_dict(), 'bt_res50_512_lambda1e3_finetuned.pth')
    print("✅ Fine‑tune complete. Model saved to bt_res50_512_lambda1e3_finetuned.pth")

if __name__ == '__main__':
    main()

