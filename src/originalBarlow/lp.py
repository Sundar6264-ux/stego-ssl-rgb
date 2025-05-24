import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import torchvision.models as models

# ─── Helpers ──────────────────────────────────────────────────────────────────
def build_encoder():
    res = models.resnet50(zero_init_residual=True)
    enc = nn.Sequential(
        res.conv1, res.bn1, res.relu,
        res.maxpool, res.layer1, res.layer2,
        res.layer3, res.layer4, res.avgpool
    )
    feat_dim = res.fc.in_features
    return enc, feat_dim

def get_loaders(batch_size=256, workers=4):
    tf_train = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    tf_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    train_ds = CIFAR10('./data', train=True,  download=True, transform=tf_train)
    test_ds  = CIFAR10('./data', train=False, download=True, transform=tf_test)
    return (
      DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                 num_workers=workers, pin_memory=True),
      DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                 num_workers=workers, pin_memory=True)
    )

# ─── Linear‐probe training ───────────────────────────────────────────────────
def linear_probe(encoder_ckpt, device='cuda'):
    # 1) build & load encoder
    enc, feat_dim = build_encoder()
    state = torch.load(encoder_ckpt, map_location='cpu')
    # strip out only the encoder part
    enc_state = {k.replace('encoder.', ''):v for k,v in state.items() if k.startswith('encoder.')}
    enc.load_state_dict(enc_state)
    enc = enc.to(device).eval()
    for p in enc.parameters(): p.requires_grad = False

    # 2) dataloaders
    train_loader, test_loader = get_loaders()

    # 3) build head
    head = nn.Linear(feat_dim, 10).to(device)
    optim_head = optim.SGD(head.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    sched = CosineAnnealingLR(optim_head, T_max=100)
    criterion = nn.CrossEntropyLoss()

    # 4) train head for 100 epochs
    for epoch in range(1, 101):
        head.train()
        total_loss = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feat = enc(x).flatten(1)
            logits = head(feat)
            loss = criterion(logits, y)
            optim_head.zero_grad()
            loss.backward()
            optim_head.step()
            total_loss += loss.item() * x.size(0)
            total      += x.size(0)
        sched.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/100: train loss {total_loss/total:.4f}")

    # 5) evaluate
    head.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            feat = enc(x).flatten(1)
            preds = head(feat).argmax(dim=1)
            correct += (preds==y).sum().item()
            total   += x.size(0)
    print(f"Linear‐probe Acc (SSL only): {100*correct/total:.2f}%")

if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    linear_probe('bt_res50_512_lambda1e3.pth', device)

