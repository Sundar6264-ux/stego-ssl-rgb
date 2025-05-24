#!/usr/bin/env python3
"""
Fine‑tune Barlow‑Twins ResNet‑50 on CIFAR‑10 with:
  • MixUp (α=0.2) + label smoothing (0.1)
  • Deeper MLP head (2048→512→10)
  • Lighter RandAugment + optional Cutout + weak RandomErasing
  • AdamW optimizer with 5‑epoch warm‑up + cosine LR decay
  • Weight decay = 5e‑4 (default)
  • AMP mixed‑precision (torch.amp)
  • cuDNN autotuner

Evaluates every N epochs:
  • Linear‑head top‑1
  • 20‑NN probe + per‑class recall
  • Top‑5 most‑confused pairs
  • t‑SNE snapshot (tsne_epoch_XX.png)
"""

import argparse, os, numpy as np, torch
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# speed
cudnn.benchmark = True

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam

def build_encoder(backbone='resnet50'):
    import torchvision.models as models
    resnet = getattr(models, backbone)(zero_init_residual=True)
    enc = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu,
        resnet.maxpool,
        resnet.layer1, resnet.layer2,
        resnet.layer3, resnet.layer4,
        resnet.avgpool
    )
    return enc, resnet.fc.in_features

def load_into_encoder(enc: nn.Module, ckpt: str, min_match=0.9):
    raw = torch.load(ckpt, map_location='cpu')
    for k in ('state_dict','model_state_dict','model'):
        if isinstance(raw, dict) and k in raw:
            raw = raw[k]; break
    raw = {k.replace('module.','').replace('encoder.',''):v
           for k,v in raw.items()}
    variants = [
        raw,
        {k.partition('.')[2]:v for k,v in raw.items() if k.split('.')[0]=='0'}
    ]
    mapping = {
      'conv1.':'0.','bn1.':'1.','relu.':'2.','maxpool.':'3.',
      'layer1.':'4.','layer2.':'5.','layer3.':'6.','layer4.':'7.',
      'avgpool.':'8.'
    }
    seq = {}
    for k,v in raw.items():
        for old,new in mapping.items():
            if k.startswith(old):
                seq[new+k[len(old):]] = v
                break
    variants.append(seq)
    valid = enc.state_dict().keys()
    variants.append({k:v for k,v in raw.items() if k in valid})
    for cand in variants:
        res = enc.load_state_dict(cand, strict=False)
        frac = 1 - len(res.missing_keys)/len(enc.state_dict())
        if frac>=min_match:
            print(f"✓ encoder weights loaded ({frac*100:.1f}% match)")
            return
    raise RuntimeError(f"Could not load ≥{min_match*100:.0f}% of encoder params")

def get_loaders(bs=256, workers=8):
    mean,std = (0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)
    def cutout(img, size=8):
        _,h,w = img.shape
        y,x = torch.randint(0,h-size,()), torch.randint(0,w-size,())
        img[:,y:y+size,x:x+size] = 0
        return img
    train_tf = T.Compose([
        T.RandomCrop(32,padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=3,magnitude=15),
        T.ToTensor(),
        T.Lambda(cutout),                    # optional Cutout
        T.RandomErasing(p=0.25,scale=(0.02,0.1),ratio=(0.3,2.0)),
        T.Normalize(mean,std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
    tr = CIFAR10('./data',train=True,transform=train_tf,download=True)
    te = CIFAR10('./data',train=False,transform=test_tf,download=True)
    return (DataLoader(tr,bs,shuffle=True, num_workers=workers,pin_memory=True),
            DataLoader(te,bs,shuffle=False,num_workers=workers,pin_memory=True))

@torch.no_grad()
def embed(enc, loader, device):
    enc.eval().to(device)
    feats, labs = [],[]
    for x,y in loader:
        x = x.to(device,non_blocking=True)
        with autocast('cuda'):
            h = enc(x).flatten(1).cpu()
        feats.append(h); labs.append(y)
    return torch.cat(feats), torch.cat(labs)

def save_tsne(emb, lbl, ep):
    pts = TSNE(2,init='pca',learning_rate='auto').fit_transform(emb.numpy())
    plt.figure(figsize=(8,8))
    for c in range(10):
        idx = (lbl==c).numpy()
        plt.scatter(pts[idx,0],pts[idx,1],s=5,label=str(c))
    plt.legend(markerscale=3,title='Class')
    plt.title(f't-SNE epoch {ep}')
    fn = f'tsne_epoch_{ep:02d}.png'
    plt.tight_layout(); plt.savefig(fn,dpi=150); plt.close()
    print(f"  ↳ saved {fn}")

@torch.no_grad()
def evaluate(model, enc, test_loader, train_loader, device, ep):
    model.eval()
    total=correct=0
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        with autocast('cuda'):
            pred = model(x).argmax(1)
        correct += (pred==y).sum().item(); total += y.size(0)
    print(f"\n🔸 epoch {ep}: linear‑head acc = {100*correct/total:.2f}%")

    tr_emb,tr_lbl = embed(enc, train_loader, device)
    te_emb,te_lbl = embed(enc, test_loader,  device)
    knn = KNeighborsClassifier(n_neighbors=20,metric='cosine',n_jobs=-1)
    knn.fit(tr_emb,tr_lbl)
    p = knn.predict(te_emb)
    print(f"   20‑NN top‑1 = {(p==te_lbl.numpy()).mean()*100:.2f}%")

    cm = confusion_matrix(te_lbl,p,labels=list(range(10)))
    for c,r in enumerate(cm.diagonal()/cm.sum(1)):
        print(f"     class {c}: recall {r*100:5.1f}%")
    counts={}
    for t,pred in zip(te_lbl.numpy(),p):
        if t!=pred:
            pair=tuple(sorted((t,pred)))
            counts[pair]=counts.get(pair,0)+1
    print("   most‑confused pairs:")
    for (a,b),n in sorted(counts.items(),key=lambda x:x[1],reverse=True)[:5]:
        print(f"     {a} ↔ {b}: {n} errors")
    save_tsne(te_emb,te_lbl,ep)

def finetune(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc,feat = build_encoder()
    load_into_encoder(enc, args.checkpoint)

    head = nn.Sequential(
        nn.Linear(feat,2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
        nn.Linear(2048,512), nn.ReLU(inplace=True),
        nn.Linear(512,10)
    )
    model = nn.Sequential(enc, nn.Flatten(), head).to(device)

    tr_loader, te_loader = get_loaders(args.batch_size, args.workers)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warmup = LambdaLR(opt, lr_lambda=lambda ep: min((ep+1)/5,1.0))
    cosine = CosineAnnealingLR(opt, T_max=args.epochs-5)
    scaler = GradScaler('cuda')
    loss_fn= nn.CrossEntropyLoss(label_smoothing=0.1)

    for ep in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(tr_loader,desc=f'epoch {ep}/{args.epochs}',ncols=80)
        for x,y in pbar:
            x, ya, yb, lam = mixup_data(x.to(device), y.to(device), alpha=0.2)
            opt.zero_grad()
            with autocast('cuda'):
                logits = model(x)
                loss   = lam*loss_fn(logits,ya)+(1-lam)*loss_fn(logits,yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=f'{loss.item():.3f}')
        # schedulers
        if ep<=5: warmup.step()
        else:     cosine.step()

        if ep % args.eval_interval == 0 or ep==args.epochs:
            evaluate(model, enc, te_loader, tr_loader, device, ep)

    out = f'finetuned_ep{args.epochs}.pth'
    torch.save(model.state_dict(), out)
    print(f"\n✓ saved {out}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',    required=True)
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--lr',            type=float, default=2e-4)
    p.add_argument('--wd',            type=float, default=5e-4)
    p.add_argument('--batch_size',    type=int,   default=256)
    p.add_argument('--workers',       type=int,   default=12)
    p.add_argument('--eval_interval', type=int,   default=20)
    return p.parse_args()

if __name__=='__main__':
    finetune(parse_args())

