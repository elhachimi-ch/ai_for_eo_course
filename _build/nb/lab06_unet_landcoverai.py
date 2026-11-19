# %% [markdown]
# # Lab 6 — U-Net semantic segmentation on LandCover.ai
# **AI for Earth Observation** · Part V · Chapter V.2
#
# LandCover.ai (Boguszewski et al., 2021): 25–50 cm aerial orthophotos of Poland with 5 classes —
# background, building, woodland, water, road. TorchGeo downloads it and cuts 512×512 tiles.
#
# We train a U-Net with an ImageNet ResNet-34 encoder (segmentation-models-pytorch), using cross-entropy + Dice loss, and report per-class IoU.
#
# Runtime: **GPU**. Download ≈ 1.5 GB.

# %%
!pip -q install torchgeo segmentation-models-pytorch opencv-python-headless

# %%
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchgeo.datasets import LandCoverAI
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["background", "building", "woodland", "water", "road"]

# %%
train_ds = LandCoverAI("data", split="train", download=True)
val_ds = LandCoverAI("data", split="val", download=True)
print(len(train_ds), len(val_ds))
s = train_ds[0]; print(s["image"].shape, s["mask"].shape, s["mask"].unique())

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def prep(batch, train=False):
    x = batch["image"].float() / 255.0
    y = batch["mask"].long()
    if train:                                   # flips: cheap, label-preserving augmentation
        if torch.rand(1) < 0.5: x, y = x.flip(-1), y.flip(-1)
        if torch.rand(1) < 0.5: x, y = x.flip(-2), y.flip(-2)
    return ((x - MEAN) / STD).to(device), y.to(device)

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=8, num_workers=2)

# %%
model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=5).to(device)
ce, dice = nn.CrossEntropyLoss(), smp.losses.DiceLoss("multiclass")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

@torch.no_grad()
def evaluate(max_batches=None):
    model.eval(); stats = []
    for i, b in enumerate(val_dl):
        x, y = prep(b)
        pred = model(x).argmax(1)
        stats.append(smp.metrics.get_stats(pred, y, mode="multiclass", num_classes=5))
        if max_batches and i >= max_batches: break
    tp, fp, fn, tn = [torch.cat([s[k] for s in stats]) for k in range(4)]
    # stats have shape (N_images, C): sum over images, IoU per class = TP / (TP + FP + FN)
    return tp.sum(0) / (tp.sum(0) + fp.sum(0) + fn.sum(0)).clamp(min=1)

EPOCHS = 5   # increase to 20+ for good results
for ep in range(EPOCHS):
    model.train()
    for i, b in enumerate(train_dl):
        x, y = prep(b, train=True)
        logits = model(x)
        loss = ce(logits, y) + dice(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    iou = evaluate(max_batches=60)
    print(f"epoch {ep}  loss {loss.item():.3f}  mIoU {iou.mean():.3f}  " +
          "  ".join(f"{c}:{v:.2f}" for c, v in zip(CLASSES, iou.tolist())))

# %% [markdown]
# ## Visualise predictions

# %%
b = next(iter(val_dl)); x, y = prep(b)
with torch.no_grad(): pred = model(x).argmax(1)
cmap = plt.matplotlib.colors.ListedColormap(["#FBF7F0", "#B5502E", "#0F6B5C", "#2E4BC6", "#C99A2E"])
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
for i in range(3):
    ax[i, 0].imshow(b["image"][i].permute(1, 2, 0).numpy().astype("uint8")); ax[i, 0].set_title("image")
    ax[i, 1].imshow(y[i].cpu(), cmap=cmap, vmin=0, vmax=4); ax[i, 1].set_title("label")
    ax[i, 2].imshow(pred[i].cpu(), cmap=cmap, vmin=0, vmax=4); ax[i, 2].set_title("prediction")
    for a in ax[i]: a.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Exercises
# 1. Which class has the lowest IoU? Why are buildings and roads harder than woodland?
# 2. Remove the Dice term. What happens to the small classes?
# 3. Swap the encoder for `"efficientnet-b3"` or a Transformer encoder (`"mit_b2"`).
# 4. This U-Net pattern is exactly what TerraTorch puts on top of Prithvi in Lab 11.
