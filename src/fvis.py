#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import argparse

def get_dataloaders(batch_size, num_workers):
    mean, std = [0.4914,0.4822,0.4465], [0.2470,0.2435,0.2616]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
    te = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    return tr, te

def build_res50_encoder():
    """ResNet-50 trunk up to avgpool."""
    r50 = models.resnet50(zero_init_residual=True)
    encoder = nn.Sequential(
        r50.conv1, r50.bn1, r50.relu, r50.maxpool,
        r50.layer1, r50.layer2, r50.layer3, r50.layer4,
        r50.avgpool
    )
    return encoder, r50.fc.in_features

def finetune_full(
    ssl_ckpt,
    epochs,
    batch_size,
    lr_backbone,
    lr_head,
    wd,
    device,
    num_workers=8
):
    print(f"[FULL FT] Using device: {device}")

    # 1) Build ResNet-50 encoder and load SSL weights
    encoder, feat_dim = build_res50_encoder()
    state = torch.load(ssl_ckpt, map_location=device)
    enc_state = {k.replace("encoder.", ""):v for k,v in state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state, strict=False)
    encoder.to(device)
    print(f"[FULL FT] Loaded SSL encoder from {ssl_ckpt}")

    # 2) Build classification head
    head = nn.Linear(feat_dim, 10).to(device)

    # 3) Unfreeze entire backbone
    for p in encoder.parameters():
        p.requires_grad = True

    # 4) DataLoaders
    tr, te = get_dataloaders(batch_size, num_workers)
    print(f"[FULL FT] Data sizes: train={len(tr.dataset)}, test={len(te.dataset)}")

    # 5) AdamW optimizer with separate LRs
    optimizer = optim.AdamW([
        {"params": encoder.parameters(), "lr": lr_backbone},
        {"params": head.parameters(),    "lr": lr_head}
    ], weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # 6) Train & eval
    train_losses, train_accs, test_accs, lrs = [], [], [], []
    for ep in range(1, epochs+1):
        print(f"[FULL FT] Epoch {ep}/{epochs}")
        encoder.train(); head.train()
        tloss = correct = total = 0
        for x,y in tqdm(tr, leave=False):
            x,y = x.to(device), y.to(device)
            feats  = encoder(x).flatten(1)
            logits = head(feats)
            loss   = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tloss   += loss.item()*x.size(0)
            preds   = logits.argmax(1)
            correct += (preds==y).sum().item()
            total   += x.size(0)

        train_losses.append(tloss/total)
        train_accs.append(correct/total*100)
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

        encoder.eval(); head.eval()
        correct = total = 0
        with torch.no_grad():
            for x,y in te:
                x,y = x.to(device), y.to(device)
                feats  = encoder(x).flatten(1)
                preds  = head(feats).argmax(1)
                correct += (preds==y).sum().item()
                total   += x.size(0)
        test_accs.append(correct/total*100)
        print(f"    Train Acc: {train_accs[-1]:.2f}%   Test Acc: {test_accs[-1]:.2f}%")

    # 7) Save checkpoint
    ft_path = ssl_ckpt.replace(".pth", "_ft_adam.pth")
    torch.save({"backbone": encoder.state_dict(),
                "head": head.state_dict()}, ft_path)
    print(f"[FULL FT] Saved checkpoint to {ft_path}")

    # 8) Plot metrics
    epochs_range = np.arange(1, epochs+1)
    plt.figure(); plt.plot(epochs_range, train_losses)
    plt.xlabel("Epoch"); plt.ylabel("Train Loss"); plt.title("Training Loss")
    plt.tight_layout(); plt.savefig("ft_train_loss.png"); plt.close()

    plt.figure(); plt.plot(epochs_range, train_accs, label="Train")
    plt.plot(epochs_range, test_accs,  label="Test")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()
    plt.title("Train/Test Acc"); plt.tight_layout()
    plt.savefig("ft_accuracy.png"); plt.close()

    plt.figure(); plt.plot(epochs_range, lrs)
    plt.xlabel("Epoch"); plt.ylabel("LR"); plt.title("LR Schedule")
    plt.tight_layout(); plt.savefig("ft_lr.png"); plt.close()

    # 9) t-SNE of head logits on first 2000 test samples
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x,y in te:
            x,y = x.to(device), y.to(device)
            feats  = encoder(x).flatten(1)
            logits = head(feats).cpu().numpy()
            all_logits.append(logits); all_labels.append(y.cpu().numpy())
            if sum(l.shape[0] for l in all_logits) >= 2000:
                break
    feats = np.vstack(all_logits)[:2000]
    labels= np.hstack(all_labels)[:2000]

    tsne = TSNE(n_components=2, init="pca", random_state=0)
    proj = tsne.fit_transform(feats)
    plt.figure(figsize=(8,8))
    for c in range(10):
        idx = labels==c
        plt.scatter(proj[idx,0], proj[idx,1], s=5, label=str(c))
    plt.legend(title="Class"); plt.title("t-SNE of Head Logits"); plt.tight_layout()
    plt.savefig("ft_tsne_logits.png"); plt.close()
    print("✅ Saved t-SNE plot to ft_tsne_logits.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl_ckpt",   type=str,
                        default="bt_res50_512_lambda1e3_stego.pth")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr_backbone",type=float, default=1e-5)
    parser.add_argument("--lr_head",    type=float, default=1e-4)
    parser.add_argument("--wd",         type=float, default=1e-6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetune_full(
        ssl_ckpt=args.ssl_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        wd=args.wd,
        device=device,
        num_workers=8
    )

