#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms, models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def get_test_loader(batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2470,0.2435,0.2616))
    ])
    ds = CIFAR10(root='./data', train=False, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)

def build_full_model(backbone='resnet50', feat_dim=2048):
    # trunk up to avgpool
    resnet = getattr(models, backbone)(zero_init_residual=True)
    encoder = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        resnet.avgpool
    )
    head = nn.Linear(feat_dim, 10)
    return nn.Sequential(encoder, nn.Flatten(), head)

def extract_head_feats(model, loader, device, max_samples):
    """Run images through full_model and collect the 10-d head logits."""
    model.eval()
    feats, labels = [], []
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)  # shape [B,10]
            feats.append(logits.cpu().numpy())
            labels.append(y.numpy())
            count += x.size(0)
            if count >= max_samples:
                break
    feats = np.vstack(feats)[:max_samples]
    labels = np.hstack(labels)[:max_samples]
    return feats, labels

def plot_tsne(feats, labels, title, out_file, perplexity, learning_rate, n_iter):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        init='pca',
        random_state=0
    )
    proj = tsne.fit_transform(feats)
    plt.figure(figsize=(8,8))
    for cls in np.unique(labels):
        idx = labels == cls
        plt.scatter(proj[idx,0], proj[idx,1], s=5, label=str(cls))
    plt.legend(title="Class", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to {out_file}")

def main():
    parser = argparse.ArgumentParser(
        description="t-SNE of fine-tuned head logits"
    )
    parser.add_argument(
        "--ft_ckpt", type=str, required=True,
        help="path to fine-tuned checkpoint .pth"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="batch size for feature extraction"
    )
    parser.add_argument(
        "--num_workers", type=int, default=16,
        help="num_workers for DataLoader"
    )
    parser.add_argument(
        "--max_samples", type=int, default=2000,
        help="max number of samples to run t-SNE on"
    )
    parser.add_argument(
        "--perplexity", type=float, default=30.0,
        help="t-SNE perplexity"
    )
    parser.add_argument(
        "--lr", type=float, default=200.0,
        help="t-SNE learning rate"
    )
    parser.add_argument(
        "--n_iter", type=int, default=1000,
        help="t-SNE number of iterations"
    )
    parser.add_argument(
        "--out_file", type=str, default="tsne_head_logits.png",
        help="output PNG filename"
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_test_loader(args.batch_size, args.num_workers)

    # build and load full model (encoder + head)
    # note: ResNet50 avgpool→feat_dim=2048
    full_model = build_full_model(backbone='resnet50', feat_dim=2048).to(device)
    state = torch.load(args.ft_ckpt, map_location='cpu')
    full_model.load_state_dict(state, strict=False)

    # extract head logits
    feats, labels = extract_head_feats(
        full_model, loader, device, args.max_samples
    )

    # t-SNE and plot
    plot_tsne(
        feats, labels,
        title="t-SNE: Head Logits (Adam fine-tune)",
        out_file=args.out_file,
        perplexity=args.perplexity,
        learning_rate=args.lr,
        n_iter=args.n_iter
    )

if __name__ == "__main__":
    main()

