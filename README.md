# Deep Learning for Earth Observation — a 3-hour course

**From Turing's learning machine and the first perceptron to CNNs, LSTMs, GANs, Transformers, geospatial foundation models, LLMs, RAG and LangGraph agents — with an Earth-observation application and code for every model family.**

Slides are a static website (classic-modern Moroccan design: zellige stars, horseshoe arches, Majorelle blue, saffron and terracotta) that runs on GitHub Pages with no build step and no external dependencies.

## Contents

| Part | Time | Theory | Practice / lab |
|---|---|---|---|
| Opening | 0:00 | Roadmap, AI ⊃ ML ⊃ DL ⊃ GenAI | — |
| I · History of AI | 0:05 | Turing's learning machine (1950), symbolic AI and the winters, classical ML, data × algorithms × chips | Random forest on Sentinel-2 |
| II · Learning maths | 0:25 | Tensors/data cubes, loss & ERM, gradient descent (live demo), backprop, likelihood | — |
| III · Perceptron & MLP | 0:35 | McCulloch–Pitts, Rosenblatt, convergence theorem, XOR, universal approximation | Perceptron playground · **Lab 1** |
| IV · CNNs | 0:55 | Convolution (live demo), ResNet, U-Net | **Labs 2–3** EuroSAT, LandCover.ai |
| Break | 1:20 | | |
| V · RNN & LSTM | 1:30 | BPTT, vanishing gradients, LSTM gates | **Lab 4** ETo forecasting (Marrakech) |
| VI · Generative models | 1:45 | GANs, pix2pix, CycleGAN, diffusion | **Lab 5** Sentinel-1 → Sentinel-2 |
| VII · Transformers | 1:55 | Attention (live demo), ViT, SAM, CLIP, EO transformers | **Lab 6** ViT + temporal Transformer |
| VIII · Geo foundation models | 2:10 | SSL, GeoFM generations, MAE vs JEPA, Zhu et al. (2026) 11 features, limitations, AtlasGeoFM | **Labs 7–8** Prithvi-EO-2.0, AlphaEarth |
| IX · Case study | 2:30 | Informal-settlement mapping with FMs (SDG 11.1.1, Morocco's *Villes sans bidonvilles*) | **Lab 9** three-tier FM pipeline |
| X · LLMs & agents | 2:40 | Pre-training, alignment, prompting, RAG, ReAct agents, MCP, LangGraph | **Labs 10–11** EO RAG, Sentinel-2 agent |

84 slides · 68 equations · 4 interactive demos · 11 Colab-ready notebooks · speaker notes on every slide.

## Publish on GitHub Pages

1. Create a repository (e.g. `dl4eo-course`) and push **the contents of this folder** to the `main` branch (the folder that contains `index.html`).
2. On GitHub: **Settings → Pages → Build and deployment → Deploy from a branch → `main` / `(root)` → Save**.
3. After a minute the course is live at `https://<your-user>.github.io/dl4eo-course/`.

The `.nojekyll` file tells GitHub Pages to serve every file as-is. The **Open in Colab** buttons on `notebooks/` detect your user and repository name automatically from the Pages URL (branch `main`).

> Large files: the pre-built `downloads/DL4EO_course.pdf` and `.pptx` are ~21 MB each — below GitHub's 100 MB file limit, so no Git LFS is needed.

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

## Editing the slides

Sources live in `_build/`:

```
_build/src/template.html          page shell (toolbar, contents, notes panel)
_build/src/slides/*.html          one file per part — edit content here
_build/nb/*.py                    notebook sources ("# %%" cells)
_build/build.py                   assembles index.html (KaTeX maths + Pygments code, rendered at build time)
_build/charts.py, motifs.py       SVG charts and zellige motifs in assets/img/
_build/export.py, make_pptx.js    regenerate downloads/ (PDF + PPTX)
```

```bash
pip install pygments playwright && playwright install chromium
npm install katex pptxgenjs
cd _build
python build.py                                   # -> ../index.html
python py2nb.py nb ../notebooks                   # -> ../notebooks/*.ipynb
python export.py && node make_pptx.js             # -> ../downloads/*.pdf, *.pptx
```

Slide syntax: `<section data-title=".." data-module=".." data-kind="theory|practice|lab|case" data-src="sources">` with an `<h2>` title and a `<div class="s-body">`; maths in `$$…$$` or `\( … \)`; code in `<pre class="py" data-file="name">…</pre>`; notes in `<aside class="notes">`.

## Sources

Course books: De Marchi & Mitchell, *Hands-On Neural Networks* (Packt, 2019) · Danka, *Mathematics of Machine Learning* (Packt, 2025) · Rothman, *Transformers for NLP and Computer Vision*, 3rd ed. (Packt, 2024) · Rodriguez, *Generative AI Foundations in Python* (Packt, 2024).
Book code: [hands-on-neural-networks](https://github.com/packtpublishing/hands-on-neural-networks) · [Transformers-for-NLP-and-Computer-Vision-3rd-Edition](https://github.com/Denis2054/Transformers-for-NLP-and-Computer-Vision-3rd-Edition) · [Generative-AI-Foundations-in-Python](https://github.com/PacktPublishing/Generative-AI-Foundations-in-Python).

Foundation-model part: Zhu, X. X. et al. (2026) *On the foundations of Earth foundation models*, Communications Earth & Environment, doi:10.1038/s43247-025-03127-x; and the GeoFM seminar slides by C. El Hachimi (figures from Vaswani et al. 2017, Xiong et al. 2024, Jakubik et al. 2023, Google DeepMind 2025, LeCun 2022 and Zhu et al. 2026 are reproduced with attribution for teaching).

Every content slide lists its references in its footer; key references are collected on slide 83.

## Notes on the notebooks

The notebooks were written against current library APIs (PyTorch, TorchGeo, TerraTorch, timm, Hugging Face, Earth Engine, LangChain/LangGraph) and syntax-checked, but they download data and models, so run them in Colab or a GPU machine with internet access. Lab 8–9 need an Earth Engine Cloud project; Lab 9 needs your own labels (none are openly available for Moroccan informal settlements); Labs 10–11 need either a local Ollama model or an LLM API key.
