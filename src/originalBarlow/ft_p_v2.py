#!/usr/bin/env python3
"""
Fine‑tune a Barlow‑Twins ResNet‑50 on CIFAR‑10 with:
  • MixUp (α=0.2) + label smoothing (0.1)
  • Deeper MLP head (2048→512→10)
  • Stronger RandAugment + Cutout + RandomErasing
  • Cosine LR decay
  • AMP mixed‑precision training (torch.amp)
  • cuDNN autotuner enabled

Evaluates every N epochs:
  • Linear‑head top‑1
  • 20‑NN probe + per‑class recall
  • Top‑5 most‑confused class pairs
  • t‑SNE snapshot (tsne_epoch_XX.png)
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Speed tweak: enable cuDNN autotuner for fixed-size inputs
# ─────────────────────────────────────────────────────────────────────────────
cudnn.benchmark = True

# ─────────────────────────────────────────────────────────────────────────────
# MixUp helper
# ─────────────────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

# ─────────────────────────────────────────────────────────────────────────────
# Build encoder (ResNet‑50 trunk)
# ─────────────────────────────────────────────────────────────────────────────
def build_encoder(backbone='resnet50'):
    import torchvision.models as models
    resnet = getattr(models, backbone)(zero_init_residual=True)
    encoder = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu,
        resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        resnet.avgpool
    )
    return encoder, resnet.fc.in_features

# ─────────────────────────────────────────────────────────────────────────────
# Robust weight loader (tries multiple key schemes, requires ≥90% match)
# ─────────────────────────────────────────────────────────────────────────────
def load_into_encoder(encoder: nn.Module, ckpt_path: str, min_match=0.9):
    raw = torch.load(ckpt_path, map_location='cpu')
    # unwrap common wrappers
    for key in ('state_dict', 'model_state_dict', 'model'):
        if isinstance(raw, dict) and key in raw:
            raw = raw[key]
            break
    # strip prefixes
    raw = {k.replace('module.', '').replace('encoder.', ''): v
           for k, v in raw.items()}
    # candidate variants
    variants = [
        raw,
        {k.partition('.')[2]: v for k, v in raw.items() if k.split('.')[0] == '0'}
    ]
    # map ResNet names → Sequential indices
    mapping = {
        'conv1.': '0.', 'bn1.': '1.', 'relu.': '2.', 'maxpool.': '3.',
        'layer1.': '4.', 'layer2.': '5.', 'layer3.': '6.',
        'layer4.': '7.', 'avgpool.': '8.'
    }
    seq = {}
    for k, v in raw.items():
        for old, new in mapping.items():
            if k.startswith(old):
                seq[new + k[len(old):]] = v
                break
    variants.append(seq)
    # exact-match keys
    valid = encoder.state_dict().keys()
    variants.append({k: v for k, v in raw.items() if k in valid})

    # try loading each
    for cand in variants:
        res = encoder.load_state_dict(cand, strict=False)
        frac = 1 - len(res.missing_keys) / len(encoder.state_dict())
        if frac >= min_match:
            print(f"✓ encoder weights loaded ({frac*100:.1f}% match)")
            if res.missing_keys:
                print("   missing keys:", res.missing_keys[:5], "…")
            return
    raise RuntimeError(f"Could not load ≥{min_match*100:.0f}% of encoder parameters")

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders with strong augmentation
# ─────────────────────────────────────────────────────────────────────────────
def get_loaders(batch_size=256, workers=8):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    def cutout(img, size=8):
        _, h, w = img.shape
        y = torch.randint(0, h - size, ())
        x = torch.randint(0, w - size, ())
        img[:, y:y+size, x:x+size] = 0
        return img

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=3, magnitude=15),
        T.ToTensor(),
        T.Lambda(cutout),
        T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    tr_ds = CIFAR10('./data', train=True, transform=train_tf, download=True)
    te_ds = CIFAR10('./data', train=False, transform=test_tf, download=True)
    tr_loader = DataLoader(tr_ds, batch_size, shuffle=True,
                           num_workers=workers, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size, shuffle=False,
                           num_workers=workers, pin_memory=True)
    return tr_loader, te_loader

# ─────────────────────────────────────────────────────────────────────────────
# Embedding extractor (with AMP inference)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed(encoder, loader, device):
    encoder.eval().to(device)
    feats, labs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast('cuda'):
            h = encoder(x).flatten(1).cpu()
        feats.append(h)
        labs.append(y)
    return torch.cat(feats), torch.cat(labs)

# ─────────────────────────────────────────────────────────────────────────────
# t‑SNE snapshot saver
# ─────────────────────────────────────────────────────────────────────────────
def save_tsne(emb, lbl, ep):
    pts = TSNE(n_components=2, init='pca', learning_rate='auto')\
          .fit_transform(emb.numpy())
    plt.figure(figsize=(8, 8))
    for c in range(10):
        idx = (lbl == c).numpy()
        plt.scatter(pts[idx, 0], pts[idx, 1], s=5, label=str(c))
    plt.legend(markerscale=3, title='Class')
    plt.title(f't‑SNE epoch {ep}')
    fn = f'tsne_epoch_{ep:02d}.png'
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()
    print(f"  ↳ saved {fn}")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: linear acc, k‑NN, recalls, confusion pairs, t‑SNE
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, encoder, test_loader, train_loader, device, ep):
    model.eval()
    total = correct = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with autocast('cuda'):
            preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    print(f"\n🔸 epoch {ep}: linear‑head acc = {100*correct/total:.2f}%")

    train_emb, train_lbl = embed(encoder, train_loader, device)
    test_emb,  test_lbl  = embed(encoder, test_loader,  device)
    knn = KNeighborsClassifier(n_neighbors=20, metric='cosine', n_jobs=-1)
    knn.fit(train_emb, train_lbl)
    preds = knn.predict(test_emb)
    print(f"   20‑NN top‑1 = {(preds == test_lbl.numpy()).mean()*100:.2f}%")

    cm     = confusion_matrix(test_lbl, preds, labels=list(range(10)))
    recall = cm.diagonal() / cm.sum(1)
    for c, r in enumerate(recall):
        print(f"     class {c}: recall {r*100:5.1f}%")

    pair_counts = {}
    for t, p in zip(test_lbl.numpy(), preds):
        if t != p:
            pair = tuple(sorted((t, p)))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    print("   most‑confused pairs:")
    for (a, b), n in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     {a} ↔ {b}: {n} errors")

    save_tsne(test_emb, test_lbl, ep)

# ─────────────────────────────────────────────────────────────────────────────
# Fine‑tune with AMP + MixUp
# ─────────────────────────────────────────────────────────────────────────────
def finetune(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder, feat_dim = build_encoder()
    load_into_encoder(encoder, args.checkpoint)

    head = nn.Sequential(
        nn.Linear(feat_dim, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
        nn.Linear(2048, 512),      nn.ReLU(inplace=True),
        nn.Linear(512, 10)
    )
    model = nn.Sequential(encoder, nn.Flatten(), head).to(device)

    tr_loader, te_loader = get_loaders(args.batch_size, args.workers)

    opt    = torch.optim.SGD(model.parameters(), lr=args.lr,
                             momentum=0.9, weight_decay=args.wd)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler('cuda')
    loss_fn= nn.CrossEntropyLoss(label_smoothing=0.1)

    for ep in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(tr_loader, desc=f'epoch {ep:3d}/{args.epochs}', ncols=100)
        for x, y in pbar:
            x, y_a, y_b, lam = mixup_data(x.to(device), y.to(device), alpha=0.2)
            opt.zero_grad()
            with autocast('cuda'):
                logits = model(x)
                loss   = lam*loss_fn(logits, y_a) + (1-lam)*loss_fn(logits, y_b)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=f'{loss.item():.3f}')
        sched.step()

        if ep % args.eval_interval == 0 or ep == args.epochs:
            evaluate(model, encoder, te_loader, tr_loader, device, ep)

    out = f'finetuned_ep{args.epochs}.pth'
    torch.save(model.state_dict(), out)
    print(f'\n✓ saved {out}')

# ─────────────────────────────────────────────────────────────────────────────
# Command‑line interface
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',    required=True,
                   help='path to SSL or fine‑tuned weights to start from')
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--lr',            type=float, default=2e-4)
    p.add_argument('--wd',            type=float, default=1e-4)
    p.add_argument('--batch_size',    type=int,   default=256)
    p.add_argument('--workers',       type=int,   default=12)
    p.add_argument('--eval_interval', type=int,   default=20)
    return p.parse_args()

if __name__ == '__main__':
    finetune(parse_args())

