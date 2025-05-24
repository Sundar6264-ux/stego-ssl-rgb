#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms, models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# ──────────── Helpers ────────────
def get_test_loader(batch_size=256, num_workers=8):
    tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2470,0.2435,0.2616))
    ])
    ds = CIFAR10(root='./data', train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)

def get_train_loader(batch_size=256, num_workers=8):
    tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2470,0.2435,0.2616))
    ])
    ds = CIFAR10(root='./data', train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)

def build_encoder(backbone='resnet50'):
    resnet = getattr(models, backbone)(zero_init_residual=True)
    return nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        resnet.avgpool
    )

def extract_feats(model, loader, device, max_samples=2000):
    model.eval()
    feats, labels = [], []
    cnt = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            h = model(x).flatten(1).cpu().numpy()
            feats.append(h); labels.append(y.numpy())
            cnt += h.shape[0]
            if cnt >= max_samples:
                break
    feats = np.vstack(feats)[:max_samples]
    labels = np.hstack(labels)[:max_samples]
    return feats, labels

def plot_tsne(feats, labels, title, out_fname):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    proj = tsne.fit_transform(feats)
    plt.figure(figsize=(8,8))
    for cls in range(10):
        idx = labels == cls
        plt.scatter(proj[idx,0], proj[idx,1], s=5, label=str(cls))
    plt.legend(title="Class", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.title(title)
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(out_fname, dpi=300)
    plt.close()
    print(f"Saved {out_fname}")

# ──────────── Main ────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test loader
    test_loader = get_test_loader()

    # 1) SSL-only embeddings
    encoder_ssl = build_encoder().to(device)
    ssl_ckpt = torch.load('bt_res50_512_lambda1e3_stego.pth', map_location='cpu')
    enc_state = {k.replace('encoder.', ''):v for k,v in ssl_ckpt.items() if k.startswith('encoder.')}
    encoder_ssl.load_state_dict(enc_state, strict=False)
    feats_ssl, labels = extract_feats(encoder_ssl, test_loader, device)
    plot_tsne(feats_ssl, labels, 't-SNE: SSL Encoder', 'tsne_ssl.png')

    # 2) Adam fine-tune (linear probe) on train set
    train_loader = get_train_loader()
    # build head
    resnet = models.resnet50(zero_init_residual=True)
    feat_dim = resnet.fc.in_features
    head = nn.Linear(feat_dim, 10).to(device)
    # freeze encoder
    for p in encoder_ssl.parameters():
        p.requires_grad = False
    # optimizer & criterion
    optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    # train head
    epochs = 15
    for epoch in range(1, epochs+1):
        encoder_ssl.eval(); head.train()
        total, correct, total_loss = 0, 0, 0.0
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                h = encoder_ssl(x).flatten(1)
            logits = head(h)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds==y).sum().item()
            total += x.size(0)
        acc = correct/total*100
        print(f"Adam fine-tune epoch {epoch}/{epochs} — Loss: {total_loss/total:.4f}, Acc: {acc:.2f}%")

    # 3) Extract and t-SNE visualize Adam-ft embeddings
    feats_adam, _ = extract_feats(encoder_ssl, test_loader, device)
    plot_tsne(feats_adam, labels, 't-SNE: Adam Linear Probe', 'tsne_adam_ft.png')

if __name__ == '__main__':
    main()

