# %% [markdown]
# # Lab 1 — Classical machine learning for land-cover mapping
# **AI for Earth Observation** · Part II
#
# Goals
# 1. Build a **pixel table** from one Sentinel-2 L2A date over Marrakech and the Haouz plain, with labels from **ESA WorldCover 2021**.
# 2. Train the classical learners of Part II side by side: naive Bayes, a decision tree, k-nearest neighbours, a random forest, gradient boosting and an SVM.
# 3. Score them two ways — **random folds** and **spatially blocked folds** — and explain the gap.
# 4. Read the confusion matrix and permutation importance, then map the whole area.
#
# Data: Microsoft Planetary Computer (no account needed) · `sentinel-2-l2a` and `esa-worldcover` (Zanaga et al., 2022).
#
# ⚠️ WorldCover is itself a model (≈ 77 % overall accuracy for 2021). Here it plays the role of reference labels: every score measures **agreement with WorldCover**, not truth on the ground.
#
# Runtime: **CPU** is fine, a few minutes — most of it downloading the scene.

# %%
!pip -q install pystac-client planetary-computer odc-stac

# %%
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import defaultdict
import pystac_client, planetary_computer, odc.stac

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
cat = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
BBOX = [-8.15, 31.35, -7.90, 31.65]            # Marrakech · Haouz plain · Lalla Takerkoust reservoir (W, S, E, N)
CRS, RES = "EPSG:32629", 20                    # UTM 29N, 20 m pixels keep the arrays small
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
VALID_SCL = [4, 5, 6, 7]                       # vegetation, not-vegetated, water, unclassified (no cloud, shadow, snow)
WC = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up", 60: "bare / sparse", 80: "water"}
WC_COL = {10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff", 50: "#fa0000", 60: "#b4b4b4", 80: "#0064c8"}


def best_day(items):
    """All tiles of the acquisition day that covers the box with the most tiles and the least cloud."""
    by_day = defaultdict(list)
    for it in items:
        by_day[it.datetime.date()].append(it)
    day = min(by_day, key=lambda d: (-len(by_day[d]), np.mean([i.properties["eo:cloud_cover"] for i in by_day[d]])))
    return by_day[day]


def load_s2(start, end, bbox=BBOX, bands=BANDS):
    """Reflectance (y, x, band) of the best day in [start, end], its clear-sky mask and the xarray grid."""
    found = cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{start}/{end}",
                       query={"eo:cloud_cover": {"lt": 10}}).item_collection()
    items = best_day(found)
    ds = odc.stac.load(items, bands=bands + ["SCL"], bbox=bbox, crs=CRS, resolution=RES,
                       groupby="solar_day", pool=16).isel(time=0)          # 16 parallel reads
    # processing baseline >= 04.00 (from 25 Jan 2022) adds a +1000 offset to the digital numbers
    offset = 1000 if float(items[0].properties.get("s2:processing_baseline", "0")) >= 4.0 else 0
    dn = np.stack([ds[b].values for b in bands], axis=-1).astype("float32")
    refl = np.clip((dn - offset) / 10_000, 0, 1.5)
    valid = np.isin(ds["SCL"].values, VALID_SCL) & np.isfinite(dn).all(-1) & (dn[..., 0] > 0)   # clear, inside the tiles
    print(f"{items[0].datetime.date()} · {len(items)} tile(s) · baseline offset {offset} · "
          f"{refl.shape[0]}×{refl.shape[1]} px · {valid.mean():.0%} clear")
    return refl, valid, ds


def rgb(refl, gain=3.0):
    return np.clip(refl[..., [2, 1, 0]] * gain, 0, 1)

# %% [markdown]
# ## 1 · Load one clear spring day and the WorldCover labels on the same grid
# `odc.stac.load(..., like=...)` reprojects WorldCover (10 m) onto the Sentinel-2 grid (20 m); `mode` resampling keeps the majority class.

# %%
refl, valid, ds = load_s2("2021-04-01", "2021-06-30")
wc_items = cat.search(collections=["esa-worldcover"], bbox=BBOX, datetime="2021").item_collection()
wc = odc.stac.load(wc_items, bands=["map"], like=ds, resampling="mode", pool=16).isel(time=0)["map"].values
valid &= np.isin(wc, list(WC))

fig, ax = plt.subplots(1, 2, figsize=(12, 7))
ax[0].imshow(rgb(refl)); ax[0].set_title("Sentinel-2 true colour")
lut = np.zeros((256, 3))                          # class code -> RGB colour
for k, c in WC_COL.items():
    lut[k] = [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
ax[1].imshow(lut[wc]); ax[1].set_title("ESA WorldCover 2021 (reference)")
for a in ax: a.axis("off")
plt.show()
print({WC[k]: f"{(wc[valid] == k).mean():.1%}" for k in WC})

# %% [markdown]
# ## 2 · Features, and labels the way analysts make them
# Indices are **hand-crafted features** — prior knowledge about how vegetation, water and built-up surfaces reflect light.
#
# Training labels are rarely isolated random pixels: an analyst digitises **polygons** — a field, a roof, a patch of water — and every pixel inside gets the label. We mimic that with small same-class patches (7 × 7 pixels ≈ 140 m) around random seed points.

# %%
B = {b: refl[..., i] for i, b in enumerate(BANDS)}
eps = 1e-6
ndvi = (B["B08"] - B["B04"]) / (B["B08"] + B["B04"] + eps)
ndwi = (B["B03"] - B["B08"]) / (B["B03"] + B["B08"] + eps)
ndbi = (B["B11"] - B["B08"]) / (B["B11"] + B["B08"] + eps)
FEAT = BANDS + ["NDVI", "NDWI", "NDBI"]
F_img = np.concatenate([refl, ndvi[..., None], ndwi[..., None], ndbi[..., None]], axis=-1)


def sample_polygons(F_img, wc, valid, per_class=1500, radius=3, seed=0):
    """Labels the way an analyst digitises them: small same-class patches ("polygons") of up to
    (2·radius+1)² pixels around random seed pixels, until each class has per_class pixels.
    radius=0 gives isolated random pixels instead."""
    rng = np.random.default_rng(seed)
    H, W = wc.shape
    taken = np.zeros_like(valid)
    out_r, out_c = [], []
    for k in WC:
        rows_k, cols_k = np.nonzero(valid & (wc == k))
        if len(rows_k) < 200:                         # skip classes too rare to learn
            continue
        got = 0
        for i in rng.permutation(len(rows_k)):
            r0, c0 = rows_k[i], cols_k[i]
            if taken[r0, c0]:
                continue
            r1, r2, c1, c2 = max(r0 - radius, 0), min(r0 + radius + 1, H), max(c0 - radius, 0), min(c0 + radius + 1, W)
            rr, cc = np.nonzero(valid[r1:r2, c1:c2] & (wc[r1:r2, c1:c2] == k) & ~taken[r1:r2, c1:c2])
            take = min(len(rr), per_class - got)
            rr, cc = rr[:take] + r1, cc[:take] + c1
            out_r.append(rr); out_c.append(cc); taken[rr, cc] = True
            got += take
            if got >= per_class:
                break
    r, c = np.concatenate(out_r), np.concatenate(out_c)
    return F_img[r, c], wc[r, c], r, c


F, y, rows, cols = sample_polygons(F_img, wc, valid, radius=3)   # 7 × 7-pixel patches ≈ 140 m
classes = sorted(np.unique(y))
print(F.shape, {WC[k]: int((y == k).sum()) for k in classes})

# %% [markdown]
# ## 3 · Six classical learners, two ways of scoring them
# **Random folds** shuffle pixels, so a test pixel almost always has a near-identical neighbour in the training set.
# **Spatial blocks** (1 km × 1 km) keep every pixel of a block on the same side of the split, which is what happens when you map a new area.

# %%
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

BLOCK = 50                                            # pixels: 50 × 20 m = 1 km
blocks = (rows // BLOCK) * 10_000 + cols // BLOCK

models = {
    "naive Bayes": GaussianNB(),
    "decision tree": DecisionTreeClassifier(min_samples_leaf=5, random_state=0),
    "k-NN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "random forest": RandomForestClassifier(300, n_jobs=-1, random_state=0),
    "gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
}
cvs = {"random folds": KFold(5, shuffle=True, random_state=0), "spatial blocks": GroupKFold(5)}
res = {}
for name, m in models.items():
    res[name] = {cv_name: cross_val_score(m, F, y, groups=blocks, cv=cv, scoring="f1_macro", n_jobs=-1).mean()
                 for cv_name, cv in cvs.items()}
res = pd.DataFrame(res).T
res["optimism"] = res["random folds"] - res["spatial blocks"]
print(res.round(3).sort_values("spatial blocks", ascending=False))

ax = res[["random folds", "spatial blocks"]].plot.barh(figsize=(8, 4), color=["#B5502E", "#0F6B5C"])
ax.set_xlabel("macro-F1"); ax.set_xlim(0, 1); ax.set_title("Same models, two ways of scoring"); plt.show()

# %% [markdown]
# **Read the table.** Every model looks better under random folds, typically by 0.08–0.13 macro-F1: pixels of one polygon land on both sides of the split, so the test set is full of near-copies of training pixels. Flexible models (k-NN, forests, boosting) gain the most, naive Bayes the least. The **ranking** can change too — the model that wins under random folds is not necessarily the one that wins on new areas. The spatial-block score is the honest one for a map of a new area.

# %% [markdown]
# ## 4 · Where does the random forest go wrong? Confusion matrix under spatial CV

# %%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

rf = RandomForestClassifier(300, n_jobs=-1, random_state=0)
pred = cross_val_predict(rf, F, y, groups=blocks, cv=GroupKFold(5))
names = [WC[k] for k in classes]
print(classification_report(y, pred, target_names=names, digits=3))
ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=names, normalize="true",
                                        values_format=".2f", cmap="Blues", xticks_rotation=45)
plt.title("Random forest, spatial-block CV (rows sum to 1)"); plt.show()

# %% [markdown]
# ## 5 · Which features matter? Permutation importance on held-out blocks
# Scramble one feature on the test blocks and measure how much macro-F1 drops. Correlated bands share credit — a feature can look unimportant because another one carries the same information.

# %%
from sklearn.inspection import permutation_importance

tr, te = next(GroupKFold(5).split(F, y, blocks))
rf.fit(F[tr], y[tr])
pi = permutation_importance(rf, F[te], y[te], scoring="f1_macro", n_repeats=5, random_state=0, n_jobs=-1)
order = np.argsort(pi.importances_mean)
plt.figure(figsize=(7, 4.5))
plt.barh(np.array(FEAT)[order], pi.importances_mean[order], xerr=pi.importances_std[order], color="#2E4BC6")
plt.xlabel("drop in macro-F1 when the feature is scrambled"); plt.title("Permutation importance (held-out blocks)"); plt.show()

# %% [markdown]
# ## 6 · Map the whole area and compare with WorldCover

# %%
rf.fit(F, y)
m = valid.ravel()
flat = F_img.reshape(-1, F_img.shape[-1])
pred_img = np.zeros(m.size, dtype="uint8")
pred_img[m] = rf.predict(flat[m])
pred_img = pred_img.reshape(valid.shape)

fig, ax = plt.subplots(1, 2, figsize=(12, 7))
ax[0].imshow(lut[pred_img]); ax[0].set_title("Random forest (trained on 1 500 pixels per class)")
ax[1].imshow(lut[np.where(valid, wc, 0)]); ax[1].set_title("WorldCover 2021")
for a in ax: a.axis("off")
plt.show()
print(f"pixel agreement with WorldCover over clear pixels: {(pred_img[valid] == wc[valid]).mean():.1%}")

# %% [markdown]
# ## Exercises
# 1. **Isolated pixels vs polygons.** Re-sample with `radius=0` (isolated random pixels) and `radius=6` (13 × 13 patches). How does the optimism of random folds change? Why does the way labels were *collected* decide how much random folds leak?
# 2. **Block size.** Repeat section 3 with `BLOCK = 12` (≈ 250 m) and `BLOCK = 250` (5 km). Which block size is large enough to keep each polygon on one side of the split?
# 3. **No indices.** Drop NDVI, NDWI and NDBI (`F[:, :10]`). Which models suffer, which do not? (Trees can build their own band ratios only approximately.)
# 4. **Class imbalance.** Sample pixels in proportion to their area instead of 1 500 per class. Compare overall accuracy and macro-F1 — which one hides the rare classes?
# 5. **Area.** Count the predicted cropland pixels and convert to hectares. Then read the *Area estimation from a sample* slide: why is pixel counting biased, and what would you need to correct it?
# 6. **Another date.** Load an autumn date instead of spring. Which classes become harder to separate?
