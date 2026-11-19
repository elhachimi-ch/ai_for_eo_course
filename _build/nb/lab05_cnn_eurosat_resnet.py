# %% [markdown]
# # Lab 5 — CNN transfer learning on EuroSAT (ResNet-18)
# **AI for Earth Observation** · Part V · Chapter V.2
#
# 1. Fine-tune an ImageNet-pre-trained ResNet-18 on EuroSAT RGB (10 classes) in two stages: head only, then the whole network.
# 2. Compare with the spectral MLP of Lab 4.
# 3. Bonus: a 13-band ResNet-18 pre-trained on Sentinel-2 with self-supervision (SSL4EO-S12 MoCo weights, via TorchGeo).
#
# Runtime: **GPU** (Colab → Runtime → Change runtime type → T4 GPU).

# %%
!pip -q install torchgeo timm scikit-learn

# %%
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"; print(device)

# %% [markdown]
# ## Data — two views of the same dataset (augmented for training, plain for evaluation)

# %%
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(), transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), norm])

full_train = datasets.EuroSAT("data", transform=train_tf, download=True)
full_eval = datasets.EuroSAT("data", transform=eval_tf, download=False)
classes = full_train.classes
idx = torch.randperm(len(full_train)).tolist()
n_tr, n_va = int(0.7 * len(idx)), int(0.15 * len(idx))
train_ds = Subset(full_train, idx[:n_tr])
val_ds = Subset(full_eval, idx[n_tr:n_tr + n_va])
test_ds = Subset(full_eval, idx[n_tr + n_va:])
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=128, num_workers=2)
test_dl = DataLoader(test_ds, batch_size=128, num_workers=2)
print(len(train_ds), len(val_ds), len(test_ds), classes)

# %% [markdown]
# ## Model — replace the ImageNet head (1000 classes) by a 10-class head

# %%
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate(dl):
    model.eval(); correct = n = 0; preds, ys = [], []
    for x, y in dl:
        p = model(x.to(device)).argmax(1).cpu()
        correct += (p == y).sum().item(); n += len(y); preds.append(p); ys.append(y)
    return correct / n, torch.cat(preds), torch.cat(ys)

def train(epochs, lr):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_dl))
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    for ep in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        print(f"  epoch {ep}  loss {loss.item():.3f}  val acc {evaluate(val_dl)[0]:.4f}")

# %% [markdown]
# ### Stage 1 — freeze the backbone, train the head (fast)

# %%
for n, p in model.named_parameters():
    p.requires_grad = n.startswith("fc")
train(epochs=2, lr=1e-3)

# %% [markdown]
# ### Stage 2 — unfreeze everything, smaller learning rate

# %%
for p in model.parameters():
    p.requires_grad = True
train(epochs=4, lr=1e-4)

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
acc, pred, y = evaluate(test_dl)
print(f"test accuracy: {acc:.4f}")
ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=classes).plot(xticks_rotation=60, cmap="Blues")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Bonus — 13-band Sentinel-2 backbone pre-trained with self-supervision
# TorchGeo ships ResNet-18 weights trained with MoCo on SSL4EO-S12 (all 13 Sentinel-2 bands).
# Here we use it as a **frozen feature extractor + logistic regression** (a linear probe) on EuroSAT-MS.

# %%
from torchgeo.datasets import EuroSAT as EuroSATMS
from torchgeo.models import ResNet18_Weights, resnet18
from sklearn.linear_model import LogisticRegression

w = ResNet18_Weights.SENTINEL2_ALL_MOCO
enc = resnet18(weights=w).to(device).eval()
enc.fc = nn.Identity()                                   # output 512-D features

@torch.no_grad()
def features(split, limit=None):
    ds = EuroSATMS("data_ms", split=split, download=True)
    dl = DataLoader(ds, batch_size=128, num_workers=2)
    F, Y = [], []
    for i, b in enumerate(dl):
        x = b["image"].float() / 10000.0                     # SSL4EO-S12 weights expect reflectance-like scaling
        x = nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
        F.append(enc(x.to(device)).cpu()); Y.append(b["label"])
        if limit and i >= limit: break
    return torch.cat(F).numpy(), torch.cat(Y).numpy()

Ftr, Ytr = features("train"); Fte, Yte = features("test")
probe = LogisticRegression(max_iter=3000).fit(Ftr, Ytr)
print("13-band SSL ResNet-18, linear probe test accuracy:", (probe.predict(Fte) == Yte).mean().round(4))

# %% [markdown]
# > Note: check the TorchGeo documentation of `ResNet18_Weights.SENTINEL2_ALL_MOCO` for the exact input normalisation recommended for your TorchGeo version; the simple `/10000` scaling above is a reasonable default.
#
# ## Exercises
# 1. Compare stage-1-only vs stage-1+2 accuracy. How much does unfreezing help?
# 2. Which classes remain confused? Compare with the MLP confusion matrix from Lab 4.
# 3. Replace the random split by a *spatial* split — why can't we do this easily with EuroSAT, and why would it matter for a real map?
