# %% [markdown]
# # Lab 2 — Unsupervised learning: PCA, clustering and change detection
# **AI for Earth Observation** · Part III
#
# Goals
# 1. **PCA** of ten Sentinel-2 bands: how many dimensions does a spectrum really have?
# 2. **k-means** and **Gaussian mixtures** to group pixels without any label; choose the number of groups (elbow, silhouette, BIC).
# 3. **DBSCAN** to find dense groups and outliers.
# 4. **Change detection** between two springs (2019 → 2024) with change-vector analysis and an Otsu threshold.
#
# Same area and loader as Lab 1. WorldCover is used **only after the fact**, to judge the clusters — no model sees a label.
#
# Runtime: **CPU**, a few minutes — most of it downloading three Sentinel-2 scenes.

# %%
!pip -q install pystac-client planetary-computer odc-stac scikit-image

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

# %%
refl, valid, ds = load_s2("2021-04-01", "2021-06-30")
wc_items = cat.search(collections=["esa-worldcover"], bbox=BBOX, datetime="2021").item_collection()
wc = odc.stac.load(wc_items, bands=["map"], like=ds, resampling="mode", pool=16).isel(time=0)["map"].values

X = refl[valid]                                   # (n_valid_pixels, 10)
mu, sd = X.mean(0), X.std(0)
Z = (X - mu) / sd                                 # standardise: every band gets equal weight
rng = np.random.default_rng(0)
sample = rng.choice(len(Z), size=30_000, replace=False)
print(Z.shape)

# %% [markdown]
# ## 1 · Principal component analysis
# PCA rotates the 10-D band space so that the first axis carries the most variance, the second the most of what is left, and so on.

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=6).fit(Z[sample])
print("cumulative explained variance:", pca.explained_variance_ratio_.cumsum().round(3))
P = pca.transform(Z)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for i in range(3):
    ax[0].plot(BANDS, pca.components_[i], marker="o", label=f"PC{i + 1}")
ax[0].axhline(0, color="k", lw=0.5); ax[0].legend(); ax[0].set_title("Loadings: what each component measures")
img = np.zeros(valid.shape + (3,))
pc3 = P[:, :3]; lo, hi = np.percentile(pc3, [2, 98], axis=0)
img[valid] = np.clip((pc3 - lo) / (hi - lo), 0, 1)
ax[1].imshow(img); ax[1].set_title("PC1–PC3 as RGB"); ax[1].axis("off")
plt.show()

# %% [markdown]
# Typically PC1 is overall **brightness**, PC2 contrasts **vegetation** (NIR) with soil and built-up (SWIR), PC3 picks up **water and moisture**. Three components carry most of the information in ten bands — the bands are strongly correlated.
#
# ## 2 · k-means: how many clusters?

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ks = range(2, 13)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(Z[sample])
    inertia.append(km.inertia_)
    sil.append(silhouette_score(Z[sample[:5000]], km.labels_[:5000]))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow: within-cluster sum of squares"); ax[0].set_xlabel("k")
ax[1].plot(ks, sil, "o-", color="#B5502E"); ax[1].set_title("Silhouette (higher is better)"); ax[1].set_xlabel("k")
plt.show()

# %%
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

K = 7
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Z[sample])
lab = km.predict(Z)
lab_img = np.full(valid.shape, -1); lab_img[valid] = lab

ref = wc[valid]
ok = np.isin(ref, list(WC))
print(f"ARI {adjusted_rand_score(ref[ok], lab[ok]):.3f} · NMI {normalized_mutual_info_score(ref[ok], lab[ok]):.3f}")
tab = pd.crosstab(pd.Series(lab[ok], name="cluster"), pd.Series([WC[v] for v in ref[ok]], name="WorldCover"), normalize="index")
print((100 * tab).round(0).astype(int))      # % of each cluster's pixels in each WorldCover class

fig, ax = plt.subplots(1, 2, figsize=(12, 7))
ax[0].imshow(rgb(refl)); ax[0].set_title("true colour")
ax[1].imshow(np.ma.masked_less(lab_img, 0), cmap="tab10", interpolation="nearest"); ax[1].set_title(f"k-means, k = {K} (colours are arbitrary ids)")
for a in ax: a.axis("off")
plt.show()

# %% [markdown]
# **Name the clusters** from the table. In this semi-arid landscape expect a **low** ARI (about 0.1): clusters follow brightness and greenness, while WorldCover's classes are thematic — bare soil, built-up and sparse grassland overlap spectrally, and cropland is green or harvested depending on the field. A cluster split between built-up and bare soil tells you those two are spectrally close here, not that k-means failed. This is why unsupervised maps need an analyst to name, merge and check the clusters.
#
# ## 3 · Gaussian mixtures: soft clusters and a principled k
# A GMM models each cluster as a Gaussian with its own covariance, fitted by expectation–maximisation. The **BIC** trades fit against the number of parameters.

# %%
from sklearn.mixture import GaussianMixture

sub = Z[sample[:15_000]]
fits = {k: GaussianMixture(k, covariance_type="full", random_state=0).fit(sub) for k in range(2, 13)}
bic = {k: g.bic(sub) for k, g in fits.items()}
best_k = min(bic, key=bic.get)
gmm = fits[best_k]
print("BIC chooses k =", best_k, "(the largest k tried)" if best_k == max(bic) else "")

proba = gmm.predict_proba(Z)
conf_img = np.full(valid.shape, np.nan); conf_img[valid] = proba.max(1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(list(bic), list(bic.values()), "o-"); ax[0].axvline(best_k, color="#B5502E", ls="--"); ax[0].set_title("BIC (lower is better)")
im = ax[1].imshow(conf_img, cmap="magma", vmin=0.5, vmax=1); ax[1].set_title("GMM: probability of the most likely cluster"); ax[1].axis("off")
plt.colorbar(im, ax=ax[1], fraction=0.035); plt.show()

# %% [markdown]
# Dark pixels sit **between** clusters: field edges, mixed pixels, peri-urban fringes. A hard k-means label hides that uncertainty.
#
# On hundreds of thousands of real pixels the BIC often keeps falling up to the largest k you try: real spectra are not a handful of Gaussian blobs, and with this much data every extra component "pays for itself". Look for the elbow of the BIC curve, or cap k at what you can still interpret.
#
# ## 4 · DBSCAN: dense groups and outliers

# %%
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

S = P[sample[:6000], :3]
d4 = np.sort(NearestNeighbors(n_neighbors=5).fit(S).kneighbors(S)[0][:, -1])
eps = float(np.percentile(d4, 90))                  # the "knee" of the 5-NN distance curve, roughly
db = DBSCAN(eps=eps, min_samples=5).fit(S)
n_cl = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print(f"eps = {eps:.2f} · clusters: {n_cl} · noise: {(db.labels_ == -1).mean():.1%}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(d4); ax[0].axhline(eps, color="#B5502E", ls="--"); ax[0].set_title("sorted distance to the 5th neighbour")
ax[1].scatter(S[:, 0], S[:, 1], c=db.labels_, s=2, cmap="tab10"); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
ax[1].set_title("DBSCAN in PC space (−1 = noise)"); plt.show()

# %% [markdown]
# ## 5 · Change detection without labels: 2019 → 2024
# Change-vector analysis: put both dates on the same grid, standardise each, and measure how far each pixel moved in band space. Otsu's method picks the threshold that best separates "little change" from "a lot of change".
#
# Comparing the **same season** avoids most of the seasonal (phenological) change. To keep the download short we zoom on Marrakech and its fast-growing fringe and use six bands.

# %%
from skimage.filters import threshold_otsu

CHG_BBOX = [-8.10, 31.58, -7.92, 31.70]                 # Marrakech and its fringe
CHG_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
r19, v19, _ = load_s2("2019-04-01", "2019-06-30", bbox=CHG_BBOX, bands=CHG_BANDS)
r24, v24, _ = load_s2("2024-04-01", "2024-06-30", bbox=CHG_BBOX, bands=CHG_BANDS)   # same box, CRS and pixel size → same grid
both = v19 & v24
z = lambda r: (r - r[both].mean(0)) / r[both].std(0)
mag = np.full(both.shape, np.nan)
mag[both] = np.linalg.norm(z(r24)[both] - z(r19)[both], axis=-1)
thr = threshold_otsu(mag[both])
change = both & (mag > thr)
print(f"Otsu threshold {thr:.2f} · changed: {change[both].mean():.1%} of clear pixels")

fig, ax = plt.subplots(1, 3, figsize=(16, 6))
ax[0].imshow(rgb(r19)); ax[0].set_title("spring 2019")
ax[1].imshow(rgb(r24)); ax[1].set_title("spring 2024")
ax[2].imshow(rgb(r24) * 0.5); ax[2].imshow(np.ma.masked_where(~change, change), cmap="autumn", alpha=0.9)
ax[2].set_title("change-vector magnitude above Otsu")
for a in ax: a.axis("off")
plt.show()

# %% [markdown]
# Zoom in on the flagged areas: new housing on Marrakech's edges, new or abandoned irrigated fields, reservoir level. Many flags are **not** land-cover change: the best clear days fall in June, at the cereal harvest, so a field harvested a week earlier in one year than the other changes a lot in band space. That is why the flagged fraction (about 18 % of this window, mostly field parcels) is far larger than the real land-cover change. An analyst still has to interpret the map — or compare annual composites instead of single dates.
#
# ## Exercises
# 1. **k = 12.** Do the new clusters split an existing class (e.g. two kinds of cropland) or find something new?
# 2. **PCA first.** Run k-means on the first 3 principal components instead of the 10 bands. What changes, and what does it cost?
# 3. **DBSCAN noise.** Map the pixels DBSCAN calls noise (you need to run it on all pixels of a small window). What are they?
# 4. **Seasonal trap.** Run section 5 with two dates of the **same** year (April and September). How much "change" is just phenology?
# 5. **Threshold.** Replace Otsu with the 95th percentile of the magnitude. Which choice would you defend in a report, and how would you validate it?
