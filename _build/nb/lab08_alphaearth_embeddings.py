# %% [markdown]
# # Lab 8 — AlphaEarth Foundations embeddings in Earth Engine
# **Deep Learning for Earth Observation** · Part VIII
#
# Google DeepMind's AlphaEarth Foundations model summarises one year of multi-sensor observations into a
# **64-dimensional, unit-length embedding for every 10 m pixel** (Earth Engine collection
# `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, 2017 onwards). Because the model was run for you, you can:
#
# 1. **Similarity search** — find places that look like a reference place (cosine similarity = dot product).
# 2. **Classify** with a few labels and a random forest (here: ESA WorldCover as a "teacher" to keep the lab self-contained).
# 3. **Detect change** between two years.
#
# Requirements: a Google Earth Engine account and a Cloud project (free for research/education). No GPU needed.

# %%
!pip -q install earthengine-api geemap

# %%
import ee, geemap
ee.Authenticate()
ee.Initialize(project="YOUR-GCP-PROJECT")      # <- replace

# %%
aoi = ee.Geometry.Rectangle([-7.75, 33.45, -7.45, 33.65])          # Casablanca
EMB = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
def embedding(year, region=aoi):
    return EMB.filterDate(f"{year}-01-01", f"{year + 1}-01-01").filterBounds(region).mosaic().clip(region)
emb = embedding(2024)
bands = emb.bandNames()
print(bands.size().getInfo(), "bands:", bands.getInfo()[:4], "...")

Map = geemap.Map(center=[33.56, -7.6], zoom=11)
Map.addLayer(emb, {"bands": ["A01", "A16", "A09"], "min": -0.3, "max": 0.3}, "embedding (3 of 64 axes)")
Map

# %% [markdown]
# ## 1 · Similarity search
# Pick a few reference points of a land type you know (inspect them on the basemap first).
# The example coordinates are placeholders — replace them with checked locations.

# %%
pts_ref = [[-7.552, 33.563], [-7.523, 33.571]]
refs = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(p)) for p in pts_ref])
ref = emb.reduceRegion(ee.Reducer.mean(), refs.geometry(), 10)
ref_img = ee.Image.constant(ref.values(bands)).rename(bands)
sim = emb.multiply(ref_img).reduce(ee.Reducer.sum()).rename("sim")      # vectors are unit-length

Map.addLayer(sim, {"min": 0.5, "max": 1, "palette": ["14213D", "2E4BC6", "C99A2E", "FBF7F0"]}, "similarity")
Map.addLayer(refs, {"color": "red"}, "references")
Map

# %% [markdown]
# ## 2 · A land-cover map from embeddings + random forest
# Labels: a stratified sample of **ESA WorldCover 2021** classes (used here only as a convenient teacher).
# In a real project, use your own field or photo-interpreted points.

# %%
wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("lc")
samples = emb.addBands(wc).stratifiedSample(numPoints=300, classBand="lc", region=aoi, scale=10,
                                            seed=1, geometries=True)
samples = samples.randomColumn("rnd", 7)
train, test = samples.filter(ee.Filter.lt("rnd", 0.7)), samples.filter(ee.Filter.gte("rnd", 0.7))
rf = ee.Classifier.smileRandomForest(200).train(train, "lc", bands)
cm = test.classify(rf).errorMatrix("lc", "classification")
print("overall accuracy:", round(cm.accuracy().getInfo(), 3), " kappa:", round(cm.kappa().getInfo(), 3))

lc_pred = emb.classify(rf)
Map.addLayer(lc_pred.randomVisualizer(), {}, "RF on embeddings")
Map.addLayer(wc.randomVisualizer(), {}, "WorldCover (teacher)", False)
Map

# %% [markdown]
# > Caution: random point splits overestimate accuracy (spatial autocorrelation). For a real evaluation, split by blocks or districts.
#
# ## 3 · Change detection between years
# $1-\cos(\mathbf{e}_{2018},\mathbf{e}_{2024})$ highlights places whose annual "signature" changed — e.g. new construction at the urban fringe.

# %%
e18, e24 = embedding(2018), embedding(2024)
change = ee.Image(1).subtract(e18.multiply(e24).reduce(ee.Reducer.sum())).rename("change")
Map.addLayer(change, {"min": 0, "max": 0.6, "palette": ["FBF7F0", "C99A2E", "B5502E"]}, "change 2018→2024")
Map

# %% [markdown]
# ## Exercises
# 1. Replace WorldCover by 50 points you digitise yourself (geemap draw tools → `Map.user_rois`). How many labels do you need?
# 2. Compare the RF on embeddings with an RF on a Sentinel-2 median composite (10 bands + NDVI). Which is better with 20 labels per class?
# 3. Export the change layer for your city and inspect the top-1 % pixels on VHR imagery. What kinds of change are they?
