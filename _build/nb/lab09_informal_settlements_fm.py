# %% [markdown]
# # Lab 9 — Mapping informal settlements with foundation models (case study)
# **Deep Learning for Earth Observation** · Part IX
#
# A coarse-to-fine, three-tier pipeline:
#
# | Tier | Data | Model | Output |
# |---|---|---|---|
# | **A** | AlphaEarth embeddings (10 m) + Open Buildings presence | Random forest (Earth Engine) | probability map per year |
# | **B** | VHR tiles (0.5–3 m) | Frozen **DINOv2** features + logistic regression | informal-settlement score per tile |
# | **C** | VHR tiles + masks | **LoRA**-adapted DINOv2 + light decoder; **SAM** boundary refinement | pixel masks |
#
# **You provide the labels.** There is no open label set for Moroccan informal settlements, so this notebook expects:
# * Tier A: a CSV `labels_points.csv` with columns `lon, lat, is_inf (0/1), city` (digitise on VHR imagery, e.g. with geemap draw tools).
# * Tiers B–C: a VHR GeoTIFF you are licensed to use (Pléiades, aerial orthophoto…) and a polygon layer `informal.gpkg` of informal-settlement outlines.
#
# Responsible use: maps of informal settlements can stigmatise or be misused. Work with the institutions and communities concerned and publish aggregated results.

# %%
!pip -q install earthengine-api geemap pandas geopandas rasterio shapely transformers peft scikit-learn

# %% [markdown]
# ## Tier A — Earth Engine: embeddings + building presence → random forest, leave-one-city-out

# %%
import ee, pandas as pd
ee.Authenticate(); ee.Initialize(project="YOUR-GCP-PROJECT")

pts = pd.read_csv("labels_points.csv")          # lon, lat, is_inf, city
labels = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"is_inf": int(r.is_inf), "city": r.city})
    for r in pts.itertuples()])

def embedding(year):
    return (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic())

def bld_presence(year):                          # Open Buildings 2.5D Temporal: 2016-2023
    ob = (ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1")
          .filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic())
    return (ob.select("building_presence")
              .reduceResolution(ee.Reducer.mean(), maxPixels=1024)
              .reproject(crs=embedding(year).projection().atScale(10))
              .rename("bld"))

YEAR = 2023
X = embedding(YEAR).addBands(bld_presence(YEAR))
train = X.sampleRegions(collection=labels, properties=["is_inf", "city"], scale=10, geometries=True)

# %%
results = {}
for test_city in sorted(pts.city.unique()):
    tr = train.filter(ee.Filter.neq("city", test_city))
    te = train.filter(ee.Filter.eq("city", test_city))
    rf = ee.Classifier.smileRandomForest(300).train(tr, "is_inf", X.bandNames())
    cm = te.classify(rf).errorMatrix("is_inf", "classification").getInfo()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    results[test_city] = round(f1, 3)
print("F1 of the informal class, leave-one-city-out:", results)

# %%
rf_all = (ee.Classifier.smileRandomForest(300).setOutputMode("PROBABILITY")
          .train(train, "is_inf", X.bandNames()))
prob = X.classify(rf_all).rename("p_informal")
import geemap
Map = geemap.Map(center=[33.57, -7.59], zoom=12)
Map.addLayer(prob, {"min": 0, "max": 1, "palette": ["FBF7F0", "C99A2E", "B5502E"]}, f"P(informal) {YEAR}")
Map
# Exercise: repeat for 2018 and map pixels whose probability rose above 0.5 (new informal growth?)

# %% [markdown]
# ## Tier B — frozen DINOv2 features on VHR tiles → few-shot logistic regression
# Tiles of 224×224 px are cut from your VHR image; a tile is labelled *informal* if ≥ 50 % of it lies inside an outline.

# %%
import numpy as np, torch, rasterio, geopandas as gpd
from rasterio.windows import Window
from shapely.geometry import box
from transformers import AutoImageProcessor, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
device = "cuda" if torch.cuda.is_available() else "cpu"

VHR, POLY, TILE = "vhr_city.tif", "informal.gpkg", 224
src = rasterio.open(VHR)
gdf = gpd.read_file(POLY).to_crs(src.crs)
poly = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union

tiles, labels_t, boxes_t = [], [], []
for row in range(0, src.height - TILE, TILE):
    for col in range(0, src.width - TILE, TILE):
        w = Window(col, row, TILE, TILE)
        arr = src.read([1, 2, 3], window=w)
        if (arr == 0).all(0).mean() > 0.1:          # skip no-data
            continue
        b = box(*rasterio.windows.bounds(w, src.transform))
        frac = b.intersection(poly).area / b.area
        if 0.1 < frac < 0.5:                         # ambiguous edge tiles: leave out
            continue
        tiles.append(np.moveaxis(arr, 0, -1).astype("uint8"))
        labels_t.append(int(frac >= 0.5)); boxes_t.append((row, col))
labels_t = np.array(labels_t)
print(len(tiles), "tiles;", labels_t.mean().round(3), "informal fraction")

# %%
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
enc = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

@torch.no_grad()
def embed(batch_tiles):
    out = enc(**proc(images=batch_tiles, return_tensors="pt").to(device))
    h = out.last_hidden_state                          # (B, 1 + patches, 768)
    return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu().numpy()

Z = np.concatenate([embed(tiles[i:i + 64]) for i in range(0, len(tiles), 64)])
# spatial split: top 70 % of the image for training, bottom 30 % for testing
rows = np.array([r for r, c in boxes_t]); cut = np.quantile(rows, 0.7)
tr, te = rows < cut, rows >= cut
for n_lab in [20, 50, 200, None]:                      # few-shot curve
    idx = np.where(tr)[0]
    if n_lab:
        pos, neg = idx[labels_t[idx] == 1], idx[labels_t[idx] == 0]
        idx = np.r_[np.random.choice(pos, min(n_lab // 2, len(pos)), replace=False),
                    np.random.choice(neg, min(n_lab // 2, len(neg)), replace=False)]
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Z[idx], labels_t[idx])
    f1 = classification_report(labels_t[te], clf.predict(Z[te]), output_dict=True)["1"]["f1-score"]
    print(f"{'all' if n_lab is None else n_lab:>4} labels -> test F1 (informal) = {f1:.3f}")

# %% [markdown]
# ## Tier C — LoRA fine-tuning for pixel masks, and SAM refinement
# DINOv2-base produces a 16×16 grid of patch tokens for a 224×224 tile. A light decoder upsamples them to a mask.
# LoRA adds small trainable matrices to the attention projections; the 86 M backbone weights stay frozen.

# %%
import torch.nn as nn, torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from rasterio.features import rasterize

lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "value"])
backbone = get_peft_model(AutoModel.from_pretrained("facebook/dinov2-base"), lora)
backbone.print_trainable_parameters()

class SegModel(nn.Module):
    def __init__(self, backbone, n_classes=2, d=768):
        super().__init__()
        self.backbone = backbone
        self.dec = nn.Sequential(
            nn.Conv2d(d, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear"), nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear"), nn.Conv2d(128, n_classes, 1))
    def forward(self, pixel_values):
        h = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 1:]     # drop CLS
        B, N, C = h.shape; s = int(N ** 0.5)
        fmap = h.transpose(1, 2).reshape(B, C, s, s)                              # (B, 768, 16, 16)
        return F.interpolate(self.dec(fmap), size=pixel_values.shape[-2:], mode="bilinear")

seg = SegModel(backbone).to(device)

def tile_mask(row, col):
    w = Window(col, row, TILE, TILE)
    return rasterize([(poly, 1)], out_shape=(TILE, TILE), transform=rasterio.windows.transform(w, src.transform))

masks = np.stack([tile_mask(r, c) for r, c in boxes_t])
mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1); std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
def to_tensor(idx):
    x = torch.tensor(np.stack([tiles[i] for i in idx])).permute(0, 3, 1, 2).float() / 255
    return ((x - mean) / std).to(device), torch.tensor(masks[idx]).long().to(device)

opt = torch.optim.AdamW([p for p in seg.parameters() if p.requires_grad], lr=2e-4)
tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
for ep in range(10):
    seg.train(); np.random.shuffle(tr_idx)
    for i in range(0, len(tr_idx), 8):
        x, y = to_tensor(tr_idx[i:i + 8])
        loss = F.cross_entropy(seg(x), y, weight=torch.tensor([1.0, 3.0], device=device))
        opt.zero_grad(); loss.backward(); opt.step()
    seg.eval(); inter = union = 0
    with torch.no_grad():
        for i in range(0, len(te_idx), 16):
            x, y = to_tensor(te_idx[i:i + 16]); p = seg(x).argmax(1)
            inter += ((p == 1) & (y == 1)).sum().item(); union += ((p == 1) | (y == 1)).sum().item()
    print(f"epoch {ep}  loss {loss.item():.3f}  IoU(informal) {inter / max(union, 1):.3f}")

# %% [markdown]
# ### SAM: refine the boundary of a detected settlement with a box prompt

# %%
from transformers import SamModel, SamProcessor
import matplotlib.pyplot as plt
sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()
sp = SamProcessor.from_pretrained("facebook/sam-vit-base")

k = te_idx[np.argmax(masks[te_idx].mean((1, 2)))]              # a test tile with much informal area
with torch.no_grad():
    x, _ = to_tensor([k]); coarse = seg(x).argmax(1)[0].cpu().numpy()
ys, xs = np.where(coarse == 1)
box_ = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else [0, 0, TILE, TILE]
inp = sp(tiles[k], input_boxes=[[box_]], return_tensors="pt").to(device)
with torch.no_grad():
    out = sam(**inp)
refined = sp.image_processor.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                                inp["reshaped_input_sizes"].cpu())[0][0, 0].numpy()
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(tiles[k]); ax[0].set_title("VHR tile")
ax[1].imshow(masks[k]); ax[1].set_title("reference")
ax[2].imshow(coarse); ax[2].set_title("LoRA-DINOv2")
ax[3].imshow(refined); ax[3].set_title("SAM refined (box prompt)")
for a in ax: a.axis("off")
plt.show()

# %% [markdown]
# ## Exercises
# 1. Replace DINOv2 by a remote-sensing encoder (e.g. a CROMA or Clay checkpoint) — does tile-level F1 improve, as the Rio favelas study suggests?
# 2. Leave-one-city-out for Tier B: train on Casablanca tiles, test on Salé or Marrakech.
# 3. Estimate the informal area with a stratified sample and confidence interval (Olofsson et al., 2014) instead of counting pixels.
# 4. Write a one-page data-governance note: who may see the map, at what resolution, and for which decisions?
