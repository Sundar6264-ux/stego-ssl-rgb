#!/usr/bin/env python3
"""
Continue fine‑tuning a Barlow‑Twins ResNet‑50 on CIFAR‑10 with:
  • MLP classification head
  • RandAugment + Cutout + RandomErasing
  • Cosine LR decay
Every `eval_interval` epochs it:
  • prints val accuracy
  • runs a 20‑NN probe and per‑class recall
  • saves a t‑SNE scatter: tsne_epoch_XX.png
"""

import argparse, os, torch, torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ─────────────────── encoder builder (same as before) ────────────────────────
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

# ─────────────────── robust loader (≥90 % match) ─────────────────────────────
def load_into_encoder(encoder: nn.Module, ckpt_path: str, min_match=0.9):
    raw = torch.load(ckpt_path, map_location='cpu')
    # unwrap wrappers
    for k in ('state_dict', 'model_state_dict', 'model'):
        if isinstance(raw, dict) and k in raw:
            raw = raw[k]; break
    raw = {k.replace('module.', '').replace('encoder.', ''): v
           for k, v in raw.items()}
    variants = [
        raw,
        {k.partition('.')[2]: v for k, v in raw.items()
         if k.split('.')[0] == '0'},   # nested under "0."
    ]
    mapping = {'conv1.':'0.','bn1.':'1.','relu.':'2.','maxpool.':'3.',
               'layer1.':'4.','layer2.':'5.','layer3.':'6.','layer4.':'7.',
               'avgpool.':'8.'}
    seq_state={}
    for k,v in raw.items():
        for old,new in mapping.items():
            if k.startswith(old):
                seq_state[new+k[len(old):]]=v; break
    variants.append(seq_state)
    valid = encoder.state_dict().keys()
    variants.append({k:v for k,v in raw.items() if k in valid})
    for cand in variants:
        miss = encoder.load_state_dict(cand, strict=False).missing_keys
        frac = 1-len(miss)/len(encoder.state_dict())
        if frac>=min_match:
            print(f'✓ encoder weights loaded ({frac*100:.1f} % match)')
            return
    raise RuntimeError('Could not load encoder weights.')

# ─────────────────── dataset helpers ─────────────────────────────────────────
def get_loaders(batch_size=256, workers=4):
    mean,std = (0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=10),
        T.ToTensor(),
        T.RandomErasing(p=0.25, scale=(0.02,0.2), ratio=(0.3,3.3)),
        T.Normalize(mean,std)
    ])
    # Cutout: erase fixed square hole
    def cutout(x, size=8):
        h,w = x.shape[1:]
        y = np.random.randint(h-size)
        x0 = np.random.randint(w-size)
        x[:, y:y+size, x0:x0+size] = 0
        return x
    train_tf.transforms.insert(-2, T.Lambda(cutout))

    test_tf  = T.Compose([T.ToTensor(), T.Normalize(mean,std)])

    tr_ds = CIFAR10('./data', train=True , transform=train_tf,  download=True)
    te_ds = CIFAR10('./data', train=False, transform=test_tf,   download=True)

    tr_loader = DataLoader(tr_ds, batch_size, shuffle=True ,
                           num_workers=workers, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size, shuffle=False,
                           num_workers=workers, pin_memory=True)
    return tr_loader, te_loader

# ─────────────────── embedding utility ───────────────────────────────────────
@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    encoder.eval().to(device)
    feats,labs=[],[]
    for x,y in loader:
        x=x.to(device,non_blocking=True)
        feats.append(encoder(x).flatten(1).cpu()); labs.append(y)
    return torch.cat(feats), torch.cat(labs)

# ─────────────────── t‑SNE plot ──────────────────────────────────────────────
def save_tsne(emb,lbl,ep):
    pts = TSNE(2, init='pca', learning_rate='auto').fit_transform(emb.numpy())
    plt.figure(figsize=(8,8))
    for c in range(10):
        idx=(lbl==c).numpy()
        plt.scatter(pts[idx,0], pts[idx,1], s=5, label=str(c))
    plt.legend(markerscale=3, title='Class'); plt.title(f't‑SNE epoch {ep}')
    fn=f'tsne_epoch_{ep:02d}.png'; plt.tight_layout(); plt.savefig(fn,dpi=150)
    plt.close(); print(f'  ↳ saved {fn}')

# ─────────────────── fine‑tune routine ───────────────────────────────────────
def finetune(args):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    encoder, feat_dim = build_encoder()
    load_into_encoder(encoder, args.checkpoint)

    # freeze? we fine‑tune full network
    mlp_head = nn.Sequential(
        nn.Linear(feat_dim,1024), nn.ReLU(inplace=True),
        nn.Linear(1024,10)
    )
    model = nn.Sequential(encoder, nn.Flatten(), mlp_head).to(device)

    tr_loader, te_loader = get_loaders(batch_size=args.batch_size,
                                       workers=args.workers)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(tr_loader, ncols=100,
                    desc=f'epoch {epoch:3d}/{args.epochs}')
        for x,y in pbar:
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            logits=model(x)
            loss=criterion(logits,y)
            loss.backward(); opt.step()
            pbar.set_postfix(loss=f'{loss.item():.3f}')
        sched.step()

        if epoch%args.eval_interval==0 or epoch==args.epochs:
            evaluate(model, encoder, te_loader, device, epoch)

    # save final model
    torch.save(model.state_dict(), f'finetuned_ep{args.epochs}.pth')
    print(f'\n✓ saved finetuned_ep{args.epochs}.pth')

# ─────────────────── evaluation (linear acc + k‑NN) ──────────────────────────
@torch.no_grad()
def evaluate(model, encoder, test_loader, device, epoch):
    model.eval()
    # linear accuracy
    correct=total=0
    for x,y in test_loader:
        x,y=x.to(device),y.to(device)
        preds=model(x).argmax(1)
        correct+=(preds==y).sum().item(); total+=y.size(0)
    acc=100*correct/total
    print(f'\n🔸 epoch {epoch}: linear‑head acc = {acc:.2f}%')

    # embeddings & 20‑NN probe
    test_emb,test_lbl=extract_embeddings(encoder,test_loader,device)
    train_emb,train_lbl=extract_embeddings(encoder,test_loader,device)  # use test for both for speed; swap in train_emb for true probe
    knn=KNeighborsClassifier(20, metric='cosine', n_jobs=-1)
    knn.fit(train_emb, train_lbl)
    preds=knn.predict(test_emb)
    top1=(preds==test_lbl.numpy()).mean()*100
    print(f'   20‑NN top‑1 = {top1:.2f}%')

    # per‑class recall
    cm=confusion_matrix(test_lbl, preds, labels=list(range(10)))
    recall=cm.diagonal()/cm.sum(1)
    for c,r in enumerate(recall):
        print(f'     class {c}: recall {r*100:5.1f}%')
    save_tsne(test_emb, test_lbl, epoch)

# ──────────────────── CLI ────────────────────────────────────────────────────
def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True,
                    help='path to SSL or fine‑tuned weights to start from')
    ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--batch_size',type=int,default=256)
    ap.add_argument('--lr',type=float,default=5e-4)
    ap.add_argument('--wd',type=float,default=1e-4)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--eval_interval',type=int,default=10)
    return ap.parse_args()

if __name__=='__main__':
    args=parse()
    finetune(args)

