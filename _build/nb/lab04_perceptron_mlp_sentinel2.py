# %% [markdown]
# # Lab 4 — Perceptron & MLP on Sentinel-2 spectra (EuroSAT-MS)
# **AI for Earth Observation** · Part V · Chapter V.1
#
# Goals
# 1. Implement Rosenblatt's perceptron from scratch (NumPy) and train it to separate **water** from **vegetation** using mean Red and NIR reflectance.
# 2. Watch it fail on a non-separable pair (Highway vs Industrial).
# 3. Train a **multi-layer perceptron** (PyTorch) on 13-band mean spectra to classify the 10 EuroSAT classes, and compare with a random forest.
#
# Data: EuroSAT-MS (Helber et al., 2019) — 27 000 Sentinel-2 patches, 64×64 px, 13 bands, downloaded through TorchGeo.
#
# Runtime: CPU is fine (GPU optional). First download ≈ 2 GB.

# %%
!pip -q install torchgeo scikit-learn matplotlib

# %%
import numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchgeo.datasets import EuroSAT
torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## Part A — the perceptron (1958)
# Labels are in {−1, +1}. On a mistake the weights move toward the misclassified example:
# $\mathbf{w} \leftarrow \mathbf{w} + \eta\, y\, \mathbf{x}$.

# %%
class Perceptron:
    """Rosenblatt (1958) with labels in {-1, +1}."""
    def __init__(self, n_features, lr=0.01):
        self.w = np.zeros(n_features); self.b = 0.0; self.lr = lr

    def predict(self, X):
        return np.where(X @ self.w + self.b >= 0, 1, -1)

    def fit(self, X, y, epochs=100):
        self.errors_ = []
        for _ in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                if yi * (xi @ self.w + self.b) <= 0:      # mistake (or on the boundary)
                    self.w += self.lr * yi * xi            # Rosenblatt update
                    self.b += self.lr * yi
                    errors += 1
            self.errors_.append(errors)
            if errors == 0:                                # converged: data separable
                break
        return self

# %% [markdown]
# Load only bands B04 (red) and B08 (NIR) and compute the **mean reflectance of each patch**.
# EuroSAT-MS stores digital numbers ×10 000, so dividing by 1e4 gives approximate reflectance.

# %%
ds_rn = EuroSAT("data", split="train", bands=("B04", "B08"), download=True)
print(len(ds_rn), "patches;", ds_rn.classes)

def mean_red_nir(ds, classes_pos, classes_neg, max_per_class=600):
    X, y, counts = [], [], {}
    for s in ds:
        name = ds.classes[int(s["label"])]
        if name not in classes_pos | classes_neg:
            continue
        if counts.get(name, 0) >= max_per_class:
            continue
        counts[name] = counts.get(name, 0) + 1
        X.append(s["image"].float().mean(dim=(1, 2)).numpy() / 1e4)
        y.append(1 if name in classes_pos else -1)
    return np.array(X), np.array(y)

X, y = mean_red_nir(ds_rn, {"Forest", "Pasture"}, {"River", "SeaLake"})
p = Perceptron(2, lr=0.05).fit(X, y, epochs=200)
print("epochs:", len(p.errors_), " train accuracy:", (p.predict(X) == y).mean())

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(X[y == 1, 0], X[y == 1, 1], s=8, c="#0F6B5C", label="vegetation")
ax[0].scatter(X[y == -1, 0], X[y == -1, 1], s=8, c="#2E4BC6", label="water")
xx = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
if abs(p.w[1]) > 1e-9:
    ax[0].plot(xx, -(p.w[0] * xx + p.b) / p.w[1], c="#B5502E", lw=2, label="decision line")
ax[0].set_xlabel("mean red reflectance"); ax[0].set_ylabel("mean NIR reflectance"); ax[0].legend()
ax[1].plot(p.errors_, marker="o"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mistakes")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Now a hard pair
# Highway vs Industrial differ by *spatial pattern*, not by mean spectrum. The perceptron never reaches zero mistakes.

# %%
X2, y2 = mean_red_nir(ds_rn, {"Highway"}, {"Industrial"})
p2 = Perceptron(2, lr=0.05).fit(X2, y2, epochs=200)
print("epochs:", len(p2.errors_), " final mistakes:", p2.errors_[-1], " accuracy:", (p2.predict(X2) == y2).mean())
plt.plot(p2.errors_); plt.xlabel("epoch"); plt.ylabel("mistakes"); plt.title("Not linearly separable"); plt.show()

# %% [markdown]
# ## Part B — a multi-layer perceptron on 13-band spectra
# Each patch → its 13 band means (a "pixel-like" spectrum). We use TorchGeo's predefined train / val / test splits.

# %%
def band_means(split):
    ds = EuroSAT("data", split=split, download=True)               # all 13 bands
    dl = DataLoader(ds, batch_size=256, num_workers=2)
    Xs, ys = [], []
    for batch in dl:
        Xs.append(batch["image"].float().mean(dim=(2, 3)) / 1e4)  # (B, 13)
        ys.append(batch["label"])
    return torch.cat(Xs), torch.cat(ys), ds.classes

Xtr, ytr, classes = band_means("train")
Xva, yva, _ = band_means("val")
Xte, yte, _ = band_means("test")
mu, sd = Xtr.mean(0), Xtr.std(0)                                     # statistics from TRAIN only
Ztr, Zva, Zte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
print(Ztr.shape, Zva.shape, Zte.shape, classes)

# %%
model = nn.Sequential(
    nn.Linear(13, 128), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10)).to(device)
print(sum(p.numel() for p in model.parameters()), "parameters")    # 10 698

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
train_dl = DataLoader(TensorDataset(Ztr, ytr), batch_size=128, shuffle=True)

@torch.no_grad()
def evaluate(Z, y):
    model.eval()
    pred = model(Z.to(device)).argmax(1).cpu()
    return (pred == y).float().mean().item(), pred

hist = []
for epoch in range(40):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)     # forward
        loss.backward()                   # back-propagation
        opt.step()                        # gradient step
    va, _ = evaluate(Zva, yva); hist.append(va)
    if epoch % 5 == 0 or epoch == 39:
        print(f"epoch {epoch:02d}  loss {loss.item():.3f}  val acc {va:.3f}")

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
te_acc, pred = evaluate(Zte, yte)
print(f"MLP test accuracy: {te_acc:.3f}")
print(classification_report(yte, pred, target_names=classes, digits=3))
ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=classes).plot(xticks_rotation=60, cmap="Blues")
plt.title("MLP on band-mean spectra"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Baseline: random forest with spectral indices
# Order of EuroSAT-MS bands: B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12.

# %%
from sklearn.ensemble import RandomForestClassifier
def with_indices(X):
    X = X.numpy(); B3, B4, B8, B11 = X[:, 2], X[:, 3], X[:, 7], X[:, 11]
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6); ndwi = (B3 - B8) / (B3 + B8 + 1e-6); ndbi = (B11 - B8) / (B11 + B8 + 1e-6)
    return np.c_[X, ndvi, ndwi, ndbi]
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(with_indices(Xtr), ytr.numpy())
print("Random forest test accuracy:", (rf.predict(with_indices(Xte)) == yte.numpy()).mean().round(3))

# %% [markdown]
# ## Exercises
# 1. Add NDVI/NDWI/NDBI as extra MLP inputs. Does accuracy change? Why might the MLP not need them?
# 2. Which classes are confused most? Why does that motivate convolutional networks (Lab 5)?
# 3. Replace ReLU with sigmoid in the MLP and compare the training curves.
