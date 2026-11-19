# %% [markdown]
# # Lab 8 — pix2pix: translating Sentinel-1 SAR into Sentinel-2 RGB
# **AI for Earth Observation** · Part V · Chapter V.4
#
# We build a small **paired** dataset ourselves from the Microsoft Planetary Computer:
# a cloud-free Sentinel-2 L2A scene and the closest-in-time Sentinel-1 RTC scene over the same area,
# resampled to the same 10 m grid, then cut into 256×256 chips.
#
# Model: pix2pix (Isola et al., 2017) — U-Net generator + PatchGAN discriminator, loss = cGAN + 100·L1.
#
# ⚠️ Generated optical images are *plausible*, not *measured*: never use them as observations without validation.
#
# Runtime: **GPU**.

# %%
!pip -q install pystac-client planetary-computer odc-stac rioxarray segmentation-models-pytorch scikit-image

# %%
import numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from datetime import timedelta
import pystac_client, planetary_computer, odc.stac
import segmentation_models_pytorch as smp
torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=planetary_computer.sign_inplace)
BBOX = [-8.15, 31.50, -7.85, 31.72]            # Marrakech and the Haouz plain (W, S, E, N)
CRS = "EPSG:32629"                             # UTM zone 29N

# %% [markdown]
# ## 1 · Find a clear Sentinel-2 scene and the closest Sentinel-1 scene

# %%
s2_items = cat.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime="2023-04-01/2023-10-31",
                      query={"eo:cloud_cover": {"lt": 2}}).item_collection()
s2_item = min(s2_items, key=lambda i: i.properties["eo:cloud_cover"])
t = s2_item.datetime
s1_items = cat.search(collections=["sentinel-1-rtc"], bbox=BBOX,
                      datetime=f"{(t - timedelta(days=6)).date()}/{(t + timedelta(days=6)).date()}").item_collection()
s1_item = min(s1_items, key=lambda i: abs(i.datetime - t))
print("S2:", s2_item.id, t.date(), " S1:", s1_item.id, s1_item.datetime.date())

# %% [markdown]
# ## 2 · Load both on the same 10 m grid
# * Sentinel-2 L2A digital numbers: subtract the 1000 offset for processing baseline ≥ 04.00 (scenes from 25 Jan 2022), divide by 10 000.
# * Sentinel-1 RTC: linear backscatter γ⁰ → decibels.

# %%
s2 = odc.stac.load([s2_item], bands=["B04", "B03", "B02"], bbox=BBOX, resolution=10, crs=CRS).isel(time=0)
s1 = odc.stac.load([s1_item], bands=["vv", "vh"], geobox=s2.odc.geobox).isel(time=0)

baseline = float(s2_item.properties.get("s2:processing_baseline", "0") or 0)
offset = 1000 if baseline >= 4.0 else 0
rgb = np.stack([s2[b].values.astype("float32") for b in ["B04", "B03", "B02"]])
rgb = np.clip((rgb - offset) / 10000.0, 0, 0.3) / 0.3              # [0, 1]
sar = np.stack([s1[b].values.astype("float32") for b in ["vv", "vh"]])
sar = 10 * np.log10(np.clip(sar, 1e-5, None))                     # dB
sar = (np.clip(sar, -25, 0) + 25) / 25                             # [0, 1]
rgb, sar = rgb * 2 - 1, sar * 2 - 1                                 # [-1, 1] to match tanh output
valid = np.isfinite(sar).all(0) & np.isfinite(rgb).all(0)
print(rgb.shape, sar.shape, "valid fraction:", valid.mean().round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(np.nan_to_num(sar[0]), cmap="gray"); ax[0].set_title("Sentinel-1 VV (dB, scaled)")
ax[1].imshow(np.nan_to_num((rgb.transpose(1, 2, 0) + 1) / 2)); ax[1].set_title("Sentinel-2 RGB")
plt.show()

# %% [markdown]
# ## 3 · Chips — spatial split (west part = train, east part = test)

# %%
P = 256
chips = []
H, W = valid.shape
for i in range(0, H - P + 1, P // 2):          # 50 % overlap inside each split
    for j in range(0, W - P + 1, P // 2):
        if valid[i:i + P, j:j + P].mean() > 0.98:
            chips.append((i, j))
cut = int(W * 0.75)
train_idx = [(i, j) for i, j in chips if j + P <= cut]
test_idx = [(i, j) for i, j in chips if j >= cut]
print(len(train_idx), "train chips,", len(test_idx), "test chips")

class Pairs(torch.utils.data.Dataset):
    def __init__(self, idx, aug=False): self.idx, self.aug = idx, aug
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i, j = self.idx[k]
        a = torch.from_numpy(np.nan_to_num(sar[:, i:i + P, j:j + P]))
        b = torch.from_numpy(np.nan_to_num(rgb[:, i:i + P, j:j + P]))
        if self.aug and torch.rand(1) < 0.5: a, b = a.flip(-1), b.flip(-1)
        if self.aug and torch.rand(1) < 0.5: a, b = a.flip(-2), b.flip(-2)
        return a, b

loader = torch.utils.data.DataLoader(Pairs(train_idx, aug=True), batch_size=8, shuffle=True, drop_last=True)

# %% [markdown]
# ## 4 · Generator (U-Net) and PatchGAN discriminator

# %%
G = smp.Unet("resnet34", encoder_weights=None, in_channels=2, classes=3, activation="tanh").to(device)

class PatchD(nn.Module):
    """Classifies overlapping patches of the (SAR, optical) pair as real or fake."""
    def __init__(self, c_in=2 + 3):
        super().__init__()
        def blk(i, o, norm=True):
            return [nn.Conv2d(i, o, 4, 2, 1)] + ([nn.InstanceNorm2d(o)] if norm else []) + [nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(*blk(c_in, 64, False), *blk(64, 128), *blk(128, 256), nn.Conv2d(256, 1, 4, 1, 1))
    def forward(self, s1, s2):
        return self.net(torch.cat([s1, s2], 1))

D = PatchD().to(device)
bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
optG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))

# %%
EPOCHS = 40
for ep in range(EPOCHS):
    G.train(); D.train()
    for s1b, s2b in loader:
        s1b, s2b = s1b.to(device), s2b.to(device)
        fake = G(s1b)
        # 1) discriminator: real -> 1, fake -> 0
        d_real, d_fake = D(s1b, s2b), D(s1b, fake.detach())
        loss_D = 0.5 * (bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake)))
        optD.zero_grad(); loss_D.backward(); optD.step()
        # 2) generator: fool D and stay close to the truth
        d_fake = D(s1b, fake)
        loss_G = bce(d_fake, torch.ones_like(d_fake)) + 100 * l1(fake, s2b)
        optG.zero_grad(); loss_G.backward(); optG.step()
    if ep % 5 == 0 or ep == EPOCHS - 1:
        print(f"epoch {ep:02d}  D {loss_D.item():.3f}  G {loss_G.item():.3f}")

# %% [markdown]
# ## 5 · Evaluate on the held-out east part

# %%
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
G.eval(); scores = []
test = Pairs(test_idx)
with torch.no_grad():
    for k in range(len(test)):
        a, b = test[k]
        f = G(a[None].to(device))[0].cpu().numpy()
        t_, f_ = (b.numpy() + 1) / 2, (f + 1) / 2
        scores.append((psnr(t_, f_, data_range=1), ssim(t_, f_, channel_axis=0, data_range=1)))
print("PSNR %.2f dB  SSIM %.3f" % tuple(np.mean(scores, 0)))

fig, ax = plt.subplots(3, 3, figsize=(10, 10))
for r, k in enumerate(np.linspace(0, len(test) - 1, 3).astype(int)):
    a, b = test[k]
    with torch.no_grad(): f = G(a[None].to(device))[0].cpu()
    ax[r, 0].imshow((a[0] + 1) / 2, cmap="gray"); ax[r, 0].set_title("S1 VV")
    ax[r, 1].imshow(((f + 1) / 2).permute(1, 2, 0).clamp(0, 1)); ax[r, 1].set_title("generated")
    ax[r, 2].imshow(((b + 1) / 2).permute(1, 2, 0)); ax[r, 2].set_title("real S2")
    for x in ax[r]: x.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Exercises
# 1. Set the L1 weight to 0 (pure GAN) and to 1000. What changes visually?
# 2. Add more scene pairs (other dates/places) — does the model generalise across seasons?
# 3. Where does the generator hallucinate structures? Discuss why GAN outputs must not replace measurements.
