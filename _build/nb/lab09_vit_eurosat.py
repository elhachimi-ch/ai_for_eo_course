# %% [markdown]
# # Lab 6 — Attention from scratch, a Vision Transformer, and a temporal Transformer
# **Deep Learning for Earth Observation** · Part VII
#
# 1. Implement scaled dot-product attention and check it against PyTorch.
# 2. Fine-tune a pre-trained **ViT-Small/16** (timm) on EuroSAT RGB and compare with the ResNet of Lab 2.
# 3. Train a small **temporal Transformer** on (synthetic) Sentinel-2-like pixel time series with irregular, cloudy dates.
#
# Runtime: **GPU** for part 2.

# %%
!pip -q install timm

# %%
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## 1 · Scaled dot-product attention
# $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k})V$

# %%
def attention(Q, K, V, mask=None):
    d = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    w = scores.softmax(dim=-1)            # (B, n, n) attention weights
    return w @ V, w

Q, K, V = torch.randn(2, 5, 16), torch.randn(2, 5, 16), torch.randn(2, 5, 16)
out, w = attention(Q, K, V)
ref = F.scaled_dot_product_attention(Q, K, V)
print("matches PyTorch:", torch.allclose(out, ref, atol=1e-5), " rows sum to 1:", torch.allclose(w.sum(-1), torch.ones(2, 5)))

# %% [markdown]
# ## 2 · Fine-tune ViT-Small/16 on EuroSAT
# 64×64 patches are resized to 224×224 → 14×14 = 196 patch tokens + one [CLS] token.

# %%
import timm
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

model = timm.create_model("vit_small_patch16_224.augreg_in21k", pretrained=True, num_classes=10).to(device)
cfg = timm.data.resolve_data_config({}, model=model)
train_tf = timm.data.create_transform(**cfg, is_training=True)
eval_tf = timm.data.create_transform(**cfg)

full_tr = datasets.EuroSAT("data", transform=train_tf, download=True)
full_ev = datasets.EuroSAT("data", transform=eval_tf)
idx = torch.randperm(len(full_tr), generator=torch.Generator().manual_seed(42)).tolist()
n_tr = int(0.8 * len(idx))
tr_dl = DataLoader(Subset(full_tr, idx[:n_tr]), batch_size=64, shuffle=True, num_workers=2)
te_dl = DataLoader(Subset(full_ev, idx[n_tr:]), batch_size=128, num_workers=2)

EPOCHS = 3
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
steps = EPOCHS * len(tr_dl)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-5, total_steps=steps, pct_start=0.1)   # warm-up + cosine
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
for ep in range(EPOCHS):
    model.train()
    for x, y in tr_dl:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval(); correct = n = 0
    with torch.no_grad():
        for x, y in te_dl:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); n += len(y)
    print(f"epoch {ep}  loss {loss.item():.3f}  test acc {correct / n:.4f}")

# %% [markdown]
# ### Look inside: attention of the [CLS] token in the last block

# %%
attn_maps = {}
def hook(module, inp, out):
    x = inp[0]; B, N, C = x.shape
    qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    attn_maps["w"] = ((q @ k.transpose(-2, -1)) * module.scale).softmax(-1).detach()
h = model.blocks[-1].attn.register_forward_hook(hook)
x, y = next(iter(te_dl)); model.eval()
with torch.no_grad(): model(x[:4].to(device))
h.remove()
cls_attn = attn_maps["w"][:, :, 0, 1:].mean(1).reshape(-1, 14, 14).cpu()   # average heads, CLS -> patches
mean, std = torch.tensor(cfg["mean"]).view(3, 1, 1), torch.tensor(cfg["std"]).view(3, 1, 1)
fig, ax = plt.subplots(2, 4, figsize=(12, 6))
for i in range(4):
    ax[0, i].imshow((x[i] * std + mean).permute(1, 2, 0).clamp(0, 1)); ax[0, i].set_title(full_ev.classes[y[i]])
    ax[1, i].imshow(cls_attn[i], cmap="magma"); ax[1, i].set_title("[CLS] attention")
    ax[0, i].axis("off"); ax[1, i].axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3 · A temporal Transformer for pixel time series
# Real crop-type benchmarks (e.g. BreizhCrops) need large downloads, so we simulate Sentinel-2-like NDVI/band trajectories
# with **double-logistic phenology**, irregular acquisition dates and random cloud gaps. The point is the mechanism:
# **dates are positions**, and cloudy observations are simply **masked**.

# %%
def double_logistic(t, sos, eos, peak, base=0.15):
    return base + (peak - base) * (1 / (1 + np.exp(-(t - sos) / 8)) - 1 / (1 + np.exp(-(t - eos) / 8)))

CROPS = {"wheat": (40, 140, 0.8), "barley": (30, 120, 0.7), "maize": (150, 260, 0.85),
         "olive": (0, 366, 0.45), "fallow": (60, 90, 0.3)}
N_BANDS, T_MAX = 4, 40
def simulate(n):
    X, D, M, Y = [], [], [], []
    for _ in range(n):
        c = np.random.randint(len(CROPS)); sos, eos, peak = list(CROPS.values())[c]
        sos += np.random.randn() * 10; eos += np.random.randn() * 10; peak += np.random.randn() * 0.05
        nd = np.random.randint(15, T_MAX + 1)
        doy = np.sort(np.random.choice(np.arange(1, 366), nd, replace=False))
        ndvi = double_logistic(doy, sos, eos, peak) + np.random.randn(nd) * 0.03
        red = 0.25 - 0.2 * ndvi; nir = 0.15 + 0.35 * ndvi; swir = 0.3 - 0.15 * ndvi
        x = np.stack([ndvi, red, nir, swir], 1)
        cloudy = np.random.rand(nd) < 0.25                     # 25 % of dates are cloudy
        x[cloudy] = np.random.rand(cloudy.sum(), N_BANDS) * 0.3 + 0.6   # bright, meaningless
        pad = T_MAX - nd
        X.append(np.pad(x, ((0, pad), (0, 0)))); D.append(np.pad(doy, (0, pad)))
        M.append(np.pad(cloudy, (0, pad), constant_values=True))      # True = ignore
        Y.append(c)
    return (torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(D)),
            torch.tensor(np.array(M)), torch.tensor(Y))

Xtr, Dtr, Mtr, Ytr = simulate(6000); Xte, Dte, Mte, Yte = simulate(1500)

class PixelTSFormer(nn.Module):
    """Pixel time series (T dates x C bands) -> class. Dates are encoded, cloudy dates masked."""
    def __init__(self, n_bands=N_BANDS, d=64, n_cls=len(CROPS), max_doy=366):
        super().__init__()
        self.inp = nn.Linear(n_bands, d)
        self.doy = nn.Embedding(max_doy + 1, d)
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=3)
        self.cls = nn.Linear(d, n_cls)
    def forward(self, x, doy, pad_mask):
        z = self.enc(self.inp(x) + self.doy(doy), src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).float()
        return self.cls((z * keep).sum(1) / keep.sum(1).clamp(min=1))     # masked mean pooling

tsf = PixelTSFormer().to(device)
opt = torch.optim.AdamW(tsf.parameters(), lr=1e-3)
dl = DataLoader(torch.utils.data.TensorDataset(Xtr, Dtr, Mtr, Ytr), batch_size=128, shuffle=True)
for ep in range(15):
    tsf.train()
    for x, d, m, y in dl:
        x, d, m, y = x.to(device), d.to(device), m.to(device), y.to(device)
        loss = F.cross_entropy(tsf(x, d, m), y); opt.zero_grad(); loss.backward(); opt.step()
    tsf.eval()
    with torch.no_grad():
        acc = (tsf(Xte.to(device), Dte.to(device), Mte.to(device)).argmax(1).cpu() == Yte).float().mean()
    if ep % 3 == 0 or ep == 14: print(f"epoch {ep}  loss {loss.item():.3f}  test acc {acc:.3f}")

# %% [markdown]
# ## Exercises
# 1. Set the cloud mask to all-False (feed the cloudy values to the model). How much does accuracy drop?
# 2. Replace the day-of-year embedding by plain index positions 0…T-1. Why does it hurt with irregular dates?
# 3. Compare ViT and ResNet-18 (Lab 2) at equal epochs. Which needs more data?
