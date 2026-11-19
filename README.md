# AI for Earth Observation

**From Turing's learning machine to agentic AI, in the order AI was invented: the history and the four learning paradigms, statistical machine learning, unsupervised learning and reinforcement learning, then deep learning from the first artificial neuron through CNNs, LSTMs, GANs and Transformers to self-supervised foundation models, LLMs, geospatial and tabular foundation models, trustworthy and responsible AI and agents — each part with its history, theory, live demos and an Earth-observation lab.**

Slides are a static website (classic-modern Moroccan design: zellige stars, horseshoe arches, Majorelle blue, saffron and terracotta) that runs on GitHub Pages with no build step and no external dependencies.

## Contents

| Part | Time | Theory | Practice / lab |
|---|---|---|---|
| Opening | 0:00 | Course map, learning outcomes | — |
| I · AI: history & paradigms | 0:05 | Timeline 1943–2026, Turing's learning machine (1950), symbolic AI and the winters, AI ⊃ ML ⊃ DL ⊃ GenAI, the four learning paradigms | Paradigms playground (supervised, unsupervised, self-supervised, RL on one EO problem) |
| II · Statistical ML | 0:30 | The classical-ML era; the EO data cube, loss & empirical risk, likelihood & cross-entropy, bias–variance & ridge (live); naive Bayes / MLC, decision trees, bagging & random forests, AdaBoost, gradient boosting, SVM; analysis-ready data, clouds & compositing, labels; spatial CV, rare-class metrics, area estimation, evaluation protocol | 6 learner playgrounds · **Lab 1** land cover from Sentinel-2 + WorldCover, random vs spatial-block CV |
| III · Unsupervised learning | 1:35 | The unsupervised era; k-means, unsupervised image classification, Gaussian mixtures & EM, DBSCAN, PCA | **Lab 2** PCA, k-means / GMM, DBSCAN, 2019 → 2024 change detection |
| IV · Reinforcement learning | 2:00 | The RL era; MDPs & value iteration, Q-learning, bandits; deep RL & RLHF as the bridge to Part V | **Lab 3** FAO-56 irrigation MDP, Q-learning, bandits for satellite tasking |
| Break | 2:25 | | |
| V.1 · Deep-learning foundations | 2:40 | Why deep learning took off (features, data, chips, compute); McCulloch–Pitts and the perceptron, gradient descent, MLP, back-propagation, activations, training recipe | Perceptron playground · **Lab 4** perceptron & MLP |
| V.2 · Vision & CNNs | 3:10 | Pixels & RGB, multi/hyperspectral, grayscale, thresholding, convolution, filtering, SAR speckle; convolution priors, ResNet, U-Net; tiles & normalisation; object detection; SAR–optical fusion | 7 vision demos · **Labs 5–6** EuroSAT, LandCover.ai |
| V.3 · RNN & LSTM | 4:05 | BPTT, vanishing gradients, LSTM gates | **Lab 7** ETo forecasting (Marrakech) |
| V.4 · Generative models | 4:17 | GANs, pix2pix, CycleGAN, diffusion | **Lab 8** Sentinel-1 → Sentinel-2 |
| V.5 · Transformers | 4:27 | Attention (live demo), ViT, SAM, CLIP, EO transformers | **Lab 9** ViT + temporal Transformer |
| Break | 4:42 | | |
| V.6 · Self-supervised learning & LLMs | 4:52 | The paradigms revisited, what makes a foundation model, SSL families; how LLMs are made, prompting, embeddings, RAG | **Lab 10** cited RAG over EO documents |
| V.7 · Geo foundation models | 5:12 | GeoFM generations, MAE vs JEPA, Zhu et al. (2026) 11 features, limitations, AtlasGeoFM; physics + ML; adapting FMs (embeddings, probes, LoRA, fine-tuning) | LAI-retrieval demo · **Labs 11–12** Prithvi-EO-2.0, AlphaEarth (+ a tabular FM) |
| V.8 · Tabular foundation models | 5:37 | In-context learning as Bayesian inference, synthetic vs real priors, architectures, the 2023–2026 landscape, TabArena / BeyondArena, proper scoring rules | 3 demos incl. a live leaderboard explorer |
| V.9 · Trustworthy & responsible AI | 5:57 | Calibration, ensemble uncertainty, explanations, domain shift; geographic bias, carbon cost, privacy, dual use, documentation | 6 demos |
| V.10 · Agentic AI | 6:22 | ReAct agents, tools, MCP, LangGraph | **Lab 13** Sentinel-2 agent |

142 slides · 244 equations · 43 interactive demos · 13 Colab-ready notebooks · speaker notes on every slide. Times are for the default settings (all blocks on, ≈ 6¾ hours with two breaks); see [Settings](#settings) to switch blocks off — `unsupervised` and `reinforcement` remove whole parts, `tabular_fm` a whole chapter — or to hide single slides for a given audience.

## Publish on GitHub Pages

1. Create a repository (e.g. `dl4eo-course`) and push **the contents of this folder** to the `main` branch (the folder that contains `index.html`).
2. On GitHub: **Settings → Pages → Build and deployment → Deploy from a branch → `main` / `(root)` → Save**.
3. After a minute the course is live at `https://<your-user>.github.io/dl4eo-course/`.

The `.nojekyll` file tells GitHub Pages to serve every file as-is. The **Open in Colab** buttons on `notebooks/` detect your user and repository name automatically from the Pages URL (branch `main`).

> Large files: the pre-built `downloads/DL4EO_course.pdf` and `.pptx` are ~32 MB and ~39 MB — below GitHub's 100 MB file limit, so no Git LFS is needed.

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

`_build/settings.toml` controls the build. For the complete edit and build workflow, see [Editing and building the course](#editing-and-building-the-course).

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
| `[modules] classical_ml` | `true` | Part II block: era timeline, naive Bayes / MLC, decision trees, bagging & RF, AdaBoost, gradient boosting, SVM, comparison (8 slides, 6 demos) |
| `[modules] eo_data` | `true` | Part II block: analysis-ready data and STAC code, clouds & compositing, where labels come from; Chapter V.2: tiles / padding / normalisation (4 slides, 2 demos) |
| `[modules] evaluation` | `true` | Part II block: spatial autocorrelation & cross-validation, metrics under class imbalance, stratified area estimation, evaluation protocol (4 slides, 3 demos) |
| `[modules] unsupervised` | `true` | the whole of Part III: era timeline, k-means, unsupervised classification of an image, GMM & EM, DBSCAN, PCA, Lab 2 (8 slides, 5 demos) |
| `[modules] reinforcement` | `true` | the whole of Part IV: era timeline, MDPs & value iteration, Q-learning, bandits, Lab 3, deep RL & RLHF (7 slides, 3 demos) |
| `[modules] cv_fundamentals` | `true` | Chapter V.2 block: pixels & RGB, multi/hyperspectral, grayscale, thresholding, convolution step by step, SAR speckle filtering (6 slides, 6 demos) |
| `[modules] detection` | `true` | Chapter V.2 block: detector families and losses, IoU / NMS / precision–recall / AP, oriented boxes and sliced inference (3 slides, 1 demo) |
| `[modules] fusion` | `true` | Chapter V.2 block: why fuse SAR and optical (early / mid / late), mapping under clouds (2 slides, 1 demo) |
| `[modules] physics_ml` | `true` | Chapter V.7 block: four ways to put physics into ML, emulators and hybrid LAI retrieval (2 slides, 1 demo) |
| `[modules] tabular_fm` | `true` | the whole of Chapter V.8: in-context learning as Bayesian averaging, priors and architectures, the 2023–2026 landscape, a TabArena / BeyondArena leaderboard explorer, proper scoring rules, EO practice (8 slides, 3 demos) |
| `[modules] trustworthy` | `true` | Chapter V.9 block: calibration & temperature scaling, ensemble uncertainty, explanations, domain shift (4 slides, 4 demos) |
| `[modules] responsible_ai` | `true` | Chapter V.9 block: geographic bias, carbon footprint, privacy / dual use / documentation (3 slides, 2 demos) |
| `[schedule]` | minutes per part, and per chapter of Part V (`V`, `V2` … `V10`), plus `break` and `break2` | section-slide times and the total duration (“6.7-hour course”) are computed from these; `[schedule.modules]` adds minutes when a block is on; a part or chapter switched off by a module does not count |

In slide sources, `data-lang="ar"` marks Arabic-only elements and `data-require="flag"` (or `"!flag"`) marks elements shown only when a module is on (or off); `"a,b"` needs both modules, `"a|b"` either; `{{slot:IV}}`, `{{start:IV}}`, `{{total_min}}`, `{{hours}}`, `{{duration}}`, `{{instructor}}` and `{{title}}` are filled in at build time.

## Editing and building the course

### 1. Edit the source files

Make changes to the source files, not the generated outputs:

```
_build/settings.toml              course settings (title, theme, modules, schedule, hidden slides)
_build/src/template.html          page shell (toolbar, contents, notes panel)
_build/src/slides/*.html          slide content in teaching order: 00 opening, 01–04 Parts I–IV,
                                  05–13 the Part V chapters V.1–V.10, 14 wrap-up
_build/nb/*.py                    notebook sources, using "# %%" cell markers
assets/css/style.css              site and slide styles
assets/js/                          slide engine and interactive demos
assets/img/                         images, diagrams, and motifs
```

For slides, edit the relevant `_build/src/slides/*.html` file. A slide is a `<section>` with an `<h2>` title and a `<div class="s-body">`; use `data-lang="ar"` for Arabic-only content and `data-require="module_name"` for optional module content. Maths can be written as `$$…$$` or `\( … \)`, and highlighted code as `<pre class="py|js|bash|yaml|json" data-file="filename">…</pre>`. Put speaker notes in `<aside class="notes">`.

Edit `_build/settings.toml` to change the title, instructor, theme, schedule, hidden slides, or optional modules. Edit `_build/nb/*.py` to change notebook content. The generated `index.html`, `notebooks/*.ipynb`, and files in `downloads/` are build outputs; rebuild them after changing their sources.

### 2. Install build tools (once)

The site build requires Python 3.11 or newer, Node.js, and the Python package Pygments. PDF/PPTX export additionally requires Playwright, its Chromium browser, and Pillow.

From the repository root:

```bash
python -m pip install pygments playwright pillow
python -m playwright install chromium
cd _build
npm install katex pptxgenjs
```

### 3. Build the site

From the repository root:

```bash
cd _build
python build.py
```

This writes the generated site to `index.html` in the repository root and updates themed SVGs in `assets/img/themed/`. Re-run it whenever you change slide sources or settings.

### 4. Regenerate notebooks (when notebook sources change)

From `_build/`:

```bash
python py2nb.py nb ../notebooks
```

This converts each `_build/nb/*.py` source into a corresponding `notebooks/*.ipynb` file.

### 5. Export the PDF and PowerPoint (optional)

Build the site first, then from `_build/` run:

```bash
python export.py
node make_pptx.js
```

The PDF is written to `downloads/DL4EO_course.pdf` by `export.py`; the PPTX is written to `downloads/DL4EO_course.pptx` by `make_pptx.js`. `export.py` uses Playwright's Chromium by default. To use an installed browser instead, set `EXPORT_BROWSER` to `msedge` or `chrome` before running it.

### 6. Preview locally

From the repository root, start a static server and open the URL in your browser:

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. Stop the server with Ctrl+C.

## Sources

Course books: De Marchi & Mitchell, *Hands-On Neural Networks* (Packt, 2019) · Danka, *Mathematics of Machine Learning* (Packt, 2025) · Rothman, *Transformers for NLP and Computer Vision*, 3rd ed. (Packt, 2024) · Rodriguez, *Generative AI Foundations in Python* (Packt, 2024).
Book code: [hands-on-neural-networks](https://github.com/packtpublishing/hands-on-neural-networks) · [Transformers-for-NLP-and-Computer-Vision-3rd-Edition](https://github.com/Denis2054/Transformers-for-NLP-and-Computer-Vision-3rd-Edition) · [Generative-AI-Foundations-in-Python](https://github.com/PacktPublishing/Generative-AI-Foundations-in-Python).

Foundation-model part: Zhu, X. X. et al. (2026) *On the foundations of Earth foundation models*, Communications Earth & Environment, doi:10.1038/s43247-025-03127-x; and the GeoFM seminar slides by C. El Hachimi (figures from Vaswani et al. 2017, Xiong et al. 2024, Jakubik et al. 2023, Google DeepMind 2025, LeCun 2022 and Zhu et al. 2026 are reproduced with attribution for teaching).

Every content slide lists its references in its footer; key references are collected on the “Key references” slide near the end.

## Notes on the notebooks

The notebooks were written against current library APIs (PyTorch, TorchGeo, TerraTorch, timm, Hugging Face, Earth Engine, LangChain/LangGraph) and syntax-checked, but they download data and models, so run them in Colab or a GPU machine with internet access. Lab 8 needs an Earth Engine Cloud project; Labs 9–10 need either a local Ollama model or an LLM API key.
