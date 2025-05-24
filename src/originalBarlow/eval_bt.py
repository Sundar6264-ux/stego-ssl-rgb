#!/usr/bin/env python3
"""
Evaluate a fine‑tuned Barlow‑Twins ResNet‑50 on CIFAR‑10.

Outputs
-------
• tsne_finetuned.png   – 2‑D t‑SNE of test‑set embeddings
• Console printout     – 20‑NN top‑1 accuracy

Checkpoint expected in the same directory:

    bt_res50_512_lambda1e3_finetuned.pth
"""

import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────
# 0) Build encoder
# ───────────────────────────────────────────────────────────────
def build_encoder(backbone: str = 'resnet50'):
    """Return (encoder, feature_dim) where encoder is a Sequential."""
    import torchvision.models as models
    resnet = getattr(models, backbone)(zero_init_residual=True)
    encoder = nn.Sequential(                   # indices 0‑8
        resnet.conv1, resnet.bn1, resnet.relu,
        resnet.maxpool,
        resnet.layer1, resnet.layer2,
        resnet.layer3, resnet.layer4,
        resnet.avgpool
    )
    return encoder, resnet.fc.in_features

# ───────────────────────────────────────────────────────────────
# 1) Robust loader – guarantees ≥ 90 % of params match
# ───────────────────────────────────────────────────────────────
def load_into_encoder(encoder: nn.Module,
                      ckpt_path: str,
                      min_match: float = 0.90):
    """
    Load *encoder* weights from ckpt_path regardless of naming scheme.
    Aborts if < min_match of parameters load.
    """
    raw = torch.load(ckpt_path, map_location='cpu')

    # unwrap common wrappers
    for k in ('state_dict', 'model_state_dict', 'model'):
        if isinstance(raw, dict) and k in raw:
            raw = raw[k]
            break

    # strip common prefixes
    raw = {k.replace('module.', '')
             .replace('backbone.', '')
             .replace('encoder.', ''): v
           for k, v in raw.items()}

    # candidate variants to try
    variants = []

    # A) keys already indices (e.g. "0.weight")
    variants.append(raw)

    # B) nested under top‑level "0." (Sequential saved as whole model)
    variants.append({k.partition('.')[2]: v
                     for k, v in raw.items()
                     if k.split('.')[0] == '0'})

    # C) ResNet names → Sequential indices map
    mapping = {
        'conv1.'  : '0.', 'bn1.': '1.', 'relu.': '2.', 'maxpool.':'3.',
        'layer1.': '4.', 'layer2.':'5.', 'layer3.':'6.',
        'layer4.': '7.', 'avgpool.':'8.',
    }
    seq_state = {}
    for k, v in raw.items():
        for old, new in mapping.items():
            if k.startswith(old):
                seq_state[new + k[len(old):]] = v
                break
    variants.append(seq_state)

    # D) keys that already match encoder exactly
    valid = encoder.state_dict().keys()
    variants.append({k: v for k, v in raw.items() if k in valid})

    # try each
    for i, cand in enumerate(variants, 1):
        result = encoder.load_state_dict(cand, strict=False)
        matched = 1 - len(result.missing_keys) / len(encoder.state_dict())
        print(f"variant {i}: matched {matched:.1%} of encoder weights")
        if matched >= min_match:
            if result.missing_keys:
                print("   ↳ still missing",
                      result.missing_keys[:10], "…")
            return
    raise RuntimeError("Could not load ≥ "
                       f"{min_match:.0%} of encoder parameters.")

# ───────────────────────────────────────────────────────────────
# 2) Embedding extractor
# ───────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(encoder: nn.Module,
                       loader : DataLoader,
                       device : str = 'cuda'):
    encoder.eval().to(device)
    feats, labs = [], []
    for x, y in tqdm(loader, desc='Embed', ncols=80):
        x = x.to(device, non_blocking=True)
        feats.append(encoder(x).flatten(1).cpu())
        labs.append(y)
    return torch.cat(feats), torch.cat(labs)

# ───────────────────────────────────────────────────────────────
# 3) t‑SNE plot
# ───────────────────────────────────────────────────────────────
def plot_tsne(emb, lbl, title, out_file):
    emb2d = TSNE(n_components=2, init='pca', learning_rate='auto')\
            .fit_transform(emb.numpy())
    plt.figure(figsize=(8, 8))
    for cls in range(10):
        idx = (lbl == cls).numpy()
        plt.scatter(emb2d[idx, 0], emb2d[idx, 1], s=5, label=str(cls))
    plt.legend(markerscale=3, title='Class')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Saved {out_file}")

# ───────────────────────────────────────────────────────────────
# 4) Main
# ───────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    ckpt = 'bt_res50_512_lambda1e3_finetuned.pth'
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"{ckpt} not found in current directory.")

    # build encoder and import weights
    encoder, _ = build_encoder()
    load_into_encoder(encoder, ckpt)

    # CIFAR‑10 loaders
    norm = T.Normalize((0.4914, 0.4822, 0.4465),
                       (0.2470, 0.2435, 0.2616))
    tf = T.Compose([T.ToTensor(), norm])

    train_ds = CIFAR10('./data', train=True,  transform=tf, download=True)
    test_ds  = CIFAR10('./data', train=False, transform=tf, download=True)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=4, pin_memory=True)

    # embeddings
    print('\nExtracting train embeddings …')
    train_emb, train_lbl = extract_embeddings(encoder, train_loader, device)
    print('Extracting test embeddings …')
    test_emb,  test_lbl  = extract_embeddings(encoder, test_loader,  device)

    # t‑SNE
    plot_tsne(test_emb, test_lbl,
              title='t‑SNE of Fine‑Tuned Encoder Features',
              out_file='tsne_finetuned.png')

    # k‑NN probe
    print('\nFitting 20‑NN probe …')
    knn = KNeighborsClassifier(n_neighbors=20,
                               metric='cosine',
                               n_jobs=-1)
    knn.fit(train_emb, train_lbl)
    top1 = knn.score(test_emb, test_lbl) * 100
    print(f'k‑NN (k=20) top‑1 accuracy: {top1:.2f}%')

# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()

