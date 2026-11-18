# AI for Earth Observation

**From Turing's learning machine, classical, unsupervised and reinforcement learning and the first perceptron to computer-vision fundamentals, CNNs, LSTMs, GANs, Transformers, geospatial foundation models, LLMs, RAG and LangGraph agents — with an Earth-observation application and code for every model family, and the EO practice around them: data preparation, object detection, SAR–optical fusion, honest evaluation, physics-aware ML, uncertainty, explainability and responsible use.**

Slides are a static website (classic-modern Moroccan design: zellige stars, horseshoe arches, Majorelle blue, saffron and terracotta) that runs on GitHub Pages with no build step and no external dependencies.

## Contents

| Part | Time | Theory | Practice / lab |
|---|---|---|---|
| Opening | 0:00 | Roadmap, AI ⊃ ML ⊃ DL ⊃ GenAI | — |
| I · History of AI, classical ML, unsupervised & RL | 0:05 | Turing's learning machine (1950), symbolic AI and the winters, the classical-ML era: naive Bayes / maximum likelihood, decision trees, bagging & random forests, AdaBoost, gradient boosting, SVM; unsupervised learning: k-means, unsupervised classification, GMM & EM, DBSCAN, PCA; reinforcement learning: MDPs & value iteration, Q-learning, bandits, deep RL & RLHF (all with the maths); data × algorithms × chips | 6 ML + 5 unsupervised + 3 RL live demos · random forest on Sentinel-2 |
| II · Learning maths | 1:10 | Tensors/data cubes, loss & ERM, gradient descent (live demo), backprop, likelihood | — |
| III · Perceptron & MLP | 1:20 | McCulloch–Pitts, Rosenblatt, convergence theorem, XOR, universal approximation | Perceptron playground · **Lab 1** |
| IV · Vision & CNNs | 1:40 | Vision fundamentals: pixels & RGB, multispectral vs hyperspectral, grayscale, thresholding (Otsu), convolution step by step, filtering, SAR speckle filtering (Lee); EO data preparation: analysis-ready data, clouds & compositing, tiles & normalisation, labels; convolution priors, ResNet, U-Net; object detection (anchors, IoU, NMS, AP, sliced inference); SAR–optical fusion; evaluating maps: spatial cross-validation, metrics for rare classes, stratified area estimation, an evaluation protocol | 7 vision + 2 data + 1 detection + 1 fusion + 3 evaluation demos · **Labs 2–3** EuroSAT, LandCover.ai |
| Break | 3:15 | | |
| V · RNN & LSTM | 3:25 | BPTT, vanishing gradients, LSTM gates | **Lab 4** ETo forecasting (Marrakech) |
| VI · Generative models | 3:40 | GANs, pix2pix, CycleGAN, diffusion | **Lab 5** Sentinel-1 → Sentinel-2 |
| VII · Transformers | 3:50 | Attention (live demo), ViT, SAM, CLIP, EO transformers | **Lab 6** ViT + temporal Transformer |
| VIII · Geo foundation models | 4:05 | SSL, GeoFM generations, MAE vs JEPA, Zhu et al. (2026) 11 features, limitations, AtlasGeoFM; physics + ML (emulators, hybrid retrieval, physics-informed losses); tabular foundation models (prior-fitted in-context learning, synthetic vs real priors, TabArena / BeyondArena, proper scoring rules); trustworthy EO AI: calibration, ensemble uncertainty, explanations, domain shift; responsible EO AI: geographic bias, carbon cost, privacy, dual use, documentation | LAI-retrieval demo · 3 tabular-FM demos · 4 trust demos · bias & carbon demos · **Labs 7–8** Prithvi-EO-2.0, AlphaEarth |
| IX · LLMs & agents | 5:25 | Pre-training, alignment, prompting, RAG, ReAct agents, MCP, LangGraph | **Labs 9–10** EO RAG, Sentinel-2 agent |

130 slides · 234 equations · 41 interactive demos · 10 Colab-ready notebooks · speaker notes on every slide. Times are for the default settings (all blocks on, ≈ 5¾ hours); see [Settings](#settings) to switch blocks off or hide single slides for a given audience (the 3-hour deep-learning original is every `[modules]` flag = false).

## Publish on GitHub Pages

1. Create a repository (e.g. `dl4eo-course`) and push **the contents of this folder** to the `main` branch (the folder that contains `index.html`).
2. On GitHub: **Settings → Pages → Build and deployment → Deploy from a branch → `main` / `(root)` → Save**.
3. After a minute the course is live at `https://<your-user>.github.io/dl4eo-course/`.

The `.nojekyll` file tells GitHub Pages to serve every file as-is. The **Open in Colab** buttons on `notebooks/` detect your user and repository name automatically from the Pages URL (branch `main`).

> Large files: the pre-built `downloads/DL4EO_course.pdf` and `.pptx` are ~28 MB and ~37 MB — below GitHub's 100 MB file limit, so no Git LFS is needed.

## Using the deck

| Key | Action |
|---|---|
| → / Space / PageDown | next slide |
| ← / PageUp | previous slide |
| Home / End | first / last |
| M | table of contents |
| N | speaker notes |
| F | fullscreen |
| ? | help |

Each slide has its own URL (`#/42`). Toolbar buttons: **PDF** and **PPTX** download the pre-built decks; **Code** opens the notebooks page; the printer icon opens a print view (`index.html?export`) — use the browser's *Save as PDF* with background graphics on.

The PPTX contains one full-slide image per slide (exactly as on the web, including formulas and code) with the speaker notes and sources in the notes pane, organised in sections by course part.

## Local preview

```bash
cd <this folder>
python -m http.server 8000      # then open http://localhost:8000
```

## Settings

`_build/settings.toml` controls the build — edit it, then run `python build.py` (and `export.py` + `make_pptx.js` to refresh the PDF/PPTX):

| Setting | Default | Effect |
|---|---|---|
| `arabic` | `false` | `true` adds the Arabic subtitles (title, section and closing slides, break row) and loads the Arabic web font; `false` leaves all Arabic out |
| `title` | `"AI for Earth Observation"` | course title on the title slide, browser tab, contents panel and PPTX (split over two lines at the first “ for ”) |
| `instructor` | `"Chouaib El Hachimi"` | name on the title slide and PPTX author |
| `position` | `"Postdoctoral Fellow"` | shown under the name on the title slide (`""` = none) |
| `affiliations` | two PolyU affiliations | numbered list under the name on the title slide; the name gets matching superscripts |
| `photo` | `"assets/img/instructor.jpg"` | portrait shown in the title-slide arch (path from the site root; `""` = zellige rosette) |
| `[theme] primary` / `secondary` | `"#C1272D"` / `"#006233"` (Moroccan flag red and green) | the two main accent colours; they replace the terracotta red and zellige green in the CSS, slides, live demos (`window.THEME`) and SVG figures (themed copies in `assets/img/themed/`). The original palette was `#B5502E` / `#0F6B5C` |
| `hide_pages` | `[]` | slides to leave out of the site, PDF and PPTX, e.g. `[1, 2, 14]`, `"1,2,14"` or `["20-25", "Break"]` (numbers as shown with `hide_pages = []`; ranges and exact slide titles allowed; the rest is renumbered) |
| `[modules] classical_ml` | `true` | Part I block: era timeline, naive Bayes / MLC, decision trees, bagging & RF, AdaBoost, gradient boosting, SVM, comparison (8 slides, 6 demos) |
| `[modules] unsupervised` | `true` | Part I block: era timeline, k-means, unsupervised classification of an image, GMM & EM, DBSCAN, PCA (6 slides, 5 demos) |
| `[modules] reinforcement` | `true` | Part I block: era timeline, MDPs & value iteration, Q-learning, bandits (ε-greedy / UCB1 / Thompson), deep RL & RLHF (5 slides, 3 demos) |
| `[modules] cv_fundamentals` | `true` | Part IV block: pixels & RGB, multi/hyperspectral, grayscale, thresholding, convolution step by step, SAR speckle filtering (6 slides, 6 demos) |
| `[modules] eo_data` | `true` | Part IV block: analysis-ready data and STAC code, clouds & compositing, tiles / padding / normalisation, where labels come from (4 slides, 2 demos) |
| `[modules] detection` | `true` | Part IV block: detector families and losses, IoU / NMS / precision–recall / AP, oriented boxes and sliced inference (3 slides, 1 demo) |
| `[modules] fusion` | `true` | Part IV block: why fuse SAR and optical (early / mid / late), mapping under clouds (2 slides, 1 demo) |
| `[modules] evaluation` | `true` | Part IV block: spatial autocorrelation & cross-validation, metrics under class imbalance, stratified area estimation, evaluation protocol (4 slides, 3 demos) |
| `[modules] physics_ml` | `true` | Part VIII block: four ways to put physics into ML, emulators and hybrid LAI retrieval (2 slides, 1 demo) |
| `[modules] tabular_fm` | `true` | Part VIII block, after Lab 8: tabular foundation models — in-context learning as Bayesian averaging, priors and architectures, the 2023–2026 model landscape, a TabArena / BeyondArena leaderboard explorer (snapshot 17 Nov 2026), proper scoring rules, EO practice with spatial CV (7 slides, 3 demos) |
| `[modules] trustworthy` | `true` | Part VIII block: calibration & temperature scaling, ensemble uncertainty, explanations, domain shift (4 slides, 4 demos) |
| `[modules] responsible_ai` | `true` | Part VIII block: geographic bias, carbon footprint, privacy / dual use / documentation (3 slides, 2 demos) |
| `[schedule]` | minutes per part | section-slide times, the course map and the total duration (“5.9-hour course”) are computed from these; `[schedule.modules]` adds minutes when a block is on |

In slide sources, `data-lang="ar"` marks Arabic-only elements and `data-require="flag"` (or `"!flag"`) marks elements shown only when a module is on (or off); `{{slot:IV}}`, `{{start:IV}}`, `{{total_min}}`, `{{hours}}`, `{{duration}}`, `{{instructor}}` and `{{title}}` are filled in at build time.

## Editing the slides

Sources live in `_build/`:

```
_build/settings.toml              course settings (title, Arabic, instructor, hidden slides, optional blocks, schedule)
_build/src/template.html          page shell (toolbar, contents, notes panel)
_build/src/slides/*.html          one file per part — edit content here
_build/nb/*.py                    notebook sources ("# %%" cells)
_build/build.py                   assembles index.html (KaTeX maths + Pygments code, rendered at build time)
assets/js/deck.js                 slide engine + original demos
assets/js/demos_ml.js             classical-ML demos (naive Bayes, tree, forest, AdaBoost, boosting, SVM)
assets/js/demos_vision.js         vision demos (pixels, spectra, grayscale, threshold, convolution, SAR)
assets/js/demos_unsup.js          unsupervised demos (k-means, scene clustering, GMM/EM, DBSCAN, PCA)
assets/js/demos_rl.js             RL demos (value iteration, Q-learning, bandits)
assets/js/demos_eo.js             EO-practice demos (compositing, tiling, NMS/AP, fusion, spatial CV, metrics, area)
assets/js/demos_trust.js          trust & responsibility demos (calibration, ensembles, attributions, domain shift,
                                  LAI retrieval, geographic bias, carbon)
assets/js/demos_tfm.js            tabular-FM demos (in-context learning under a prior, TabArena explorer with an
                                  embedded leaderboard snapshot, scoring rules)
_build/charts.py, motifs.py       SVG charts and zellige motifs in assets/img/
_build/export.py, make_pptx.js    regenerate downloads/ (PDF + PPTX)
```

```bash
pip install pygments playwright && playwright install chromium   # Python ≥ 3.11 (tomllib)
npm install katex pptxgenjs
cd _build
python build.py                                   # -> ../index.html
python py2nb.py nb ../notebooks                   # -> ../notebooks/*.ipynb
python export.py && node make_pptx.js             # -> ../downloads/*.pdf, *.pptx
# EXPORT_BROWSER=msedge (or chrome) python export.py   uses an installed browser instead
```

Slide syntax: `<section data-title=".." data-module=".." data-kind="theory|practice|lab|case" data-src="sources">` with an `<h2>` title and a `<div class="s-body">`; maths in `$$…$$` or `\( … \)`; code in `<pre class="py" data-file="name">…</pre>`; notes in `<aside class="notes">`.

## Sources

Course books: De Marchi & Mitchell, *Hands-On Neural Networks* (Packt, 2019) · Danka, *Mathematics of Machine Learning* (Packt, 2025) · Rothman, *Transformers for NLP and Computer Vision*, 3rd ed. (Packt, 2024) · Rodriguez, *Generative AI Foundations in Python* (Packt, 2024).
Book code: [hands-on-neural-networks](https://github.com/packtpublishing/hands-on-neural-networks) · [Transformers-for-NLP-and-Computer-Vision-3rd-Edition](https://github.com/Denis2054/Transformers-for-NLP-and-Computer-Vision-3rd-Edition) · [Generative-AI-Foundations-in-Python](https://github.com/PacktPublishing/Generative-AI-Foundations-in-Python).

Foundation-model part: Zhu, X. X. et al. (2026) *On the foundations of Earth foundation models*, Communications Earth & Environment, doi:10.1038/s43247-025-03127-x; and the GeoFM seminar slides by C. El Hachimi (figures from Vaswani et al. 2017, Xiong et al. 2024, Jakubik et al. 2023, Google DeepMind 2025, LeCun 2022 and Zhu et al. 2026 are reproduced with attribution for teaching).

Every content slide lists its references in its footer; key references are collected on the “Key references” slide near the end.

## Notes on the notebooks

The notebooks were written against current library APIs (PyTorch, TorchGeo, TerraTorch, timm, Hugging Face, Earth Engine, LangChain/LangGraph) and syntax-checked, but they download data and models, so run them in Colab or a GPU machine with internet access. Lab 8 needs an Earth Engine Cloud project; Labs 9–10 need either a local Ollama model or an LLM API key.
