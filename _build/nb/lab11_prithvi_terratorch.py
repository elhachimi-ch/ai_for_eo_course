# %% [markdown]
# # Lab 11 — Fine-tuning the Prithvi-EO-2.0 geospatial foundation model with TerraTorch
# **AI for Earth Observation** · Part V · Chapter V.7
#
# * **Model**: Prithvi-EO-2.0-300M (NASA & IBM; Szwarcman et al., 2024) — a ViT pre-trained as a spatio-temporal masked autoencoder on 4.2 M global HLS samples.
# * **Toolkit**: TerraTorch — backbone + neck + decoder + head, trained with PyTorch Lightning.
# * **Task**: burn-scar segmentation on the *HLS Burn Scars* dataset (6 HLS bands, 512×512 chips, binary masks).
#
# Runtime: **GPU** (a Colab T4 works with batch size 2–4 and mixed precision). Check the TerraTorch docs if an argument name changed in your version.

# %%
!pip -q install terratorch huggingface_hub rasterio

# %%
import glob, os, tarfile, numpy as np, torch, rasterio, matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import lightning.pytorch as pl
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask
pl.seed_everything(0)

# %% [markdown]
# ## 1 · Download the HLS Burn Scars dataset from Hugging Face

# %%
if not os.path.isdir("hls_burn_scars"):
    tgz = hf_hub_download("ibm-nasa-geospatial/hls_burn_scars", "hls_burn_scars.tar.gz", repo_type="dataset", local_dir=".")
    with tarfile.open(tgz) as t:
        t.extractall(".")
# locate training / validation folders whatever the archive layout
def find_dir(name):
    hits = [d for d in glob.glob(f"hls_burn_scars/**/{name}", recursive=True) if os.path.isdir(d)]
    return hits[0] if hits else None
TRAIN_DIR, VAL_DIR = find_dir("training"), find_dir("validation")
print(TRAIN_DIR, VAL_DIR, len(glob.glob(f"{TRAIN_DIR}/*_merged.tif")), "training images")

# %% [markdown]
# ## 2 · Band statistics from the training set (for normalisation)

# %%
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
files = sorted(glob.glob(f"{TRAIN_DIR}/*_merged.tif"))
acc = []
for f in files[::5]:                                   # every 5th image is enough for statistics
    with rasterio.open(f) as src:
        a = src.read().astype("float32").reshape(src.count, -1)
        a = a[:, (a > -9000).all(0)]                   # drop no-data
        acc.append(a[:, ::50])
acc = np.concatenate(acc, 1)
MEANS, STDS = acc.mean(1).tolist(), acc.std(1).tolist()
print("means", np.round(MEANS, 4), "\nstds ", np.round(STDS, 4))

# %% [markdown]
# ## 3 · Data module and task

# %%
dm = GenericNonGeoSegmentationDataModule(
    batch_size=4, num_workers=2, num_classes=2,
    train_data_root=TRAIN_DIR, val_data_root=VAL_DIR, test_data_root=VAL_DIR,
    img_grep="*_merged.tif", label_grep="*.mask.tif",
    means=MEANS, stds=STDS,
    dataset_bands=BANDS, output_bands=BANDS,
    no_data_replace=0, no_label_replace=-1)

model = SemanticSegmentationTask(
    model_factory="EncoderDecoderFactory",
    model_args=dict(
        backbone="prithvi_eo_v2_300", backbone_pretrained=True,
        backbone_bands=BANDS, backbone_num_frames=1,
        necks=[{"name": "SelectIndices", "indices": [5, 11, 17, 23]},   # 4 intermediate ViT blocks
               {"name": "ReshapeTokensToImage"},                         # tokens -> feature maps
               {"name": "LearnedInterpolateToPyramidal"}],               # build a multi-scale pyramid
        decoder="UNetDecoder", decoder_channels=[512, 256, 128, 64],
        head_dropout=0.1, num_classes=2),
    loss="dice", optimizer="AdamW", lr=1e-4,
    ignore_index=-1, freeze_backbone=False,
    class_names=["unburned", "burned"])

trainer = pl.Trainer(max_epochs=10, accelerator="auto", precision="16-mixed",
                     log_every_n_steps=5, default_root_dir="runs/prithvi_burnscars")
trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)

# %% [markdown]
# ## 4 · Look at predictions

# %%
dm.setup("test")
batch = next(iter(dm.test_dataloader()))
model.eval()
with torch.no_grad():
    out = model(batch["image"].to(model.device))
pred = out.output.argmax(1).cpu() if hasattr(out, "output") else out.argmax(1).cpu()
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
for i in range(3):
    img = batch["image"][i][[2, 1, 0]].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    ax[i, 0].imshow(img); ax[i, 0].set_title("HLS RGB (normalised)")
    ax[i, 1].imshow(batch["mask"][i], vmin=-1, vmax=1); ax[i, 1].set_title("label")
    ax[i, 2].imshow(pred[i], vmin=0, vmax=1); ax[i, 2].set_title("Prithvi prediction")
    for a in ax[i]: a.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Exercises
# 1. `freeze_backbone=True` — how close does a frozen Prithvi get? (≈ linear-probe regime)
# 2. Try `backbone="prithvi_eo_v2_300_tl"` (temporal + location embeddings) and pass metadata as described in the TerraTorch docs.
# 3. Compare with the U-Net of Lab 6 trained from ImageNet weights on the same data. When does the foundation model pay off? (Hint: reduce the training set to 10 % of the images.)
# 4. The same code runs from a YAML file: `terratorch fit -c config.yaml`.
