#!/usr/bin/env python3
import argparse
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torchvision.models as models
from sklearn.neighbors import KNeighborsClassifier

# ─── Model definition ─────────────────────────────────────────────────────────
class BarlowTwinsModel(nn.Module):
    def __init__(self, backbone='resnet50', proj_hidden_dim=512, out_dim=512):
        super().__init__()
        resnet = getattr(models, backbone)(zero_init_residual=True)
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool
        )
        enc_dim = resnet.fc.in_features
        self.projector = nn.Sequential(
            nn.Linear(enc_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)
        )

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        return self.projector(h)

# ─── Data loaders ────────────────────────────────────────────────────────────
def get_cifar10_loader(train: bool, batch_size=256, num_workers=16):
    tf = T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    ds = CIFAR10(root='./data', train=train, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)

# ─── SSL k-NN evaluation ──────────────────────────────────────────────────────
def eval_knn(ssl_ckpt, k=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model
    model = BarlowTwinsModel(backbone='resnet50', proj_hidden_dim=512, out_dim=512)
    state = torch.load(ssl_ckpt, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # extract embeddings
    train_loader = get_cifar10_loader(train=True)
    test_loader  = get_cifar10_loader(train=False)

    X_train, y_train = [], []
    with torch.no_grad():
        for x,y in train_loader:
            x = x.to(device)
            emb = model.encoder(x).flatten(1).cpu().numpy()
            X_train.append(emb); y_train.append(y.numpy())
    X_train = np.vstack(X_train); y_train = np.hstack(y_train)

    X_test, y_test = [], []
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            emb = model.encoder(x).flatten(1).cpu().numpy()
            X_test.append(emb); y_test.append(y.numpy())
    X_test = np.vstack(X_test); y_test = np.hstack(y_test)

    # k-NN
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test) * 100
    print(f"🧮 {k}-NN accuracy (SSL): {acc:.2f}%")

# ─── Fine-tuned evaluation ────────────────────────────────────────────────────
def eval_finetuned(ft_ckpt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # rebuild encoder + head
    resnet = models.resnet50(zero_init_residual=True)
    encoder = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        resnet.avgpool
    )
    feat_dim = resnet.fc.in_features
    head = nn.Linear(feat_dim, 10)
    model = nn.Sequential(encoder, nn.Flatten(), head)
    ck = torch.load(ft_ckpt, map_location='cpu')
    model.load_state_dict(ck, strict=False)
    model.to(device).eval()

    loader = get_cifar10_loader(train=False)
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds==y).sum().item()
            total += y.size(0)
    acc = correct/total * 100
    print(f"✅ Fine-tuned accuracy: {acc:.2f}%")

# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl_ckpt", type=str, required=True,
                        help="path to SSL checkpoint .pth")
    parser.add_argument("--ft_ckpt",  type=str, required=True,
                        help="path to fine-tuned checkpoint .pth")
    parser.add_argument("--knn_k",    type=int, default=20,
                        help="k for k-NN on SSL features")
    args = parser.parse_args()

    print("\n== k-NN eval on SSL model ==")
    eval_knn(args.ssl_ckpt, k=args.knn_k)
    print("\n== Direct eval of fine-tuned model ==")
    eval_finetuned(args.ft_ckpt)

