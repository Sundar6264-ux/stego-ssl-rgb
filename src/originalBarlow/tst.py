#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import torchvision.models as models
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────────────────
# 0) Helpers: build encoder, evaluation, embedding extraction, TSNE plotting
# ───────────────────────────────────────────────────────────────────────────────
def build_encoder(backbone='resnet50'):
    """Construct ResNet encoder (up to avgpool) and return (encoder_module, feat_dim)."""
    resnet = getattr(models, backbone)(zero_init_residual=True)
    encoder = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu,
        resnet.maxpool, resnet.layer1, resnet.layer2,
        resnet.layer3, resnet.layer4, resnet.avgpool
    )
    feat_dim = resnet.fc.in_features
    return encoder, feat_dim

def eval_checkpoint(ckpt_path, device='cuda'):
    """Load a .pth of (encoder+projector or full model) into a linear probe and eval."""
    # build encoder + head
    encoder, feat_dim = build_encoder()
    head = nn.Linear(feat_dim, 10)
    model = nn.Sequential(encoder, nn.Flatten(), head).to(device).eval()
    # load state
    state = torch.load(ckpt_path, map_location='cpu')
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # maybe state has encoder.* and projector.* keys
        # strip "encoder." prefix and ignore projector
        enc_state = {k.replace('encoder.', ''): v
                     for k,v in state.items() if k.startswith('encoder.')}
        encoder.load_state_dict(enc_state)
        # load head if present
        head_state = {k.replace('fc.', ''): v
                      for k,v in state.items() if k.startswith('fc.')}
        if head_state:
            head.load_state_dict(head_state)
    # test loader
    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.4914,0.4822,0.4465),
                                (0.2470,0.2435,0.2616))])
    ds = CIFAR10(root='./data', train=False, transform=tf, download=True)
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        num_workers=4, pin_memory=True)
    # eval
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds==y).sum().item()
            total   += y.size(0)
    acc = 100*correct/total
    print(f"{os.path.basename(ckpt_path)} → Test Acc: {acc:.2f}%")
    return acc

def extract_embeddings(encoder, loader, device='cuda'):
    """Extract feature vectors from encoder for all images in loader."""
    encoder = encoder.to(device).eval()
    feats = []; labs = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            h = encoder(x).flatten(1).cpu()
            feats.append(h); labs.append(y)
    return torch.cat(feats,0), torch.cat(labs,0)

def plot_tsne(embeddings, labels, title="t-SNE", out_file="tsne.png"):
    """Run t-SNE and plot a colored scatter of the embeddings."""
    emb2d = TSNE(n_components=2, init='pca').fit_transform(embeddings.numpy())
    plt.figure(figsize=(8,8))
    for cls in range(10):
        idxs = (labels==cls).numpy()
        plt.scatter(emb2d[idxs,0], emb2d[idxs,1], label=str(cls), s=5)
    plt.legend(markerscale=3, title="Class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Saved t-SNE plot to {out_file}")

# ───────────────────────────────────────────────────────────────────────────────
# Main script
# ───────────────────────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # 1) Evaluate both checkpoints
    ckpts = ['bt_res50_512_lambda1e3.pth',
             'bt_res50_512_lambda1e3_finetuned.pth']
    for ckpt in ckpts:
        if not os.path.isfile(ckpt):
            print(f"Checkpoint not found: {ckpt}")
            continue
        eval_checkpoint(ckpt, device)

    # 2) t-SNE of SSL encoder (pre-fine-tune)
    print("\nExtracting embeddings for t-SNE from SSL encoder...")
    encoder, feat_dim = build_encoder()
    # load only encoder weights from SSL checkpoint
    ssl_state = torch.load('bt_res50_512_lambda1e3.pth', map_location='cpu')
    enc_state = {k.replace('encoder.', ''):v
                 for k,v in ssl_state.items() if k.startswith('encoder.')}
    encoder.load_state_dict(enc_state)
    # prepare test loader
    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.4914,0.4822,0.4465),
                                (0.2470,0.2435,0.2616))])
    test_ds = CIFAR10(root='./data', train=False, transform=tf, download=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=4, pin_memory=True)
    embs, labs = extract_embeddings(encoder, test_loader, device)
    plot_tsne(embs, labs, title="t-SNE of SSL Encoder Features",
              out_file="tsne_ssl.png")

if __name__ == '__main__':
    main()

