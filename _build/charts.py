"""Hand-built SVG charts and diagrams in the course palette."""
import math, os
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "img") + os.sep
INK, MAJ, ZEL, TER, SAF, MUT, LINE = "#14213D", "#2E4BC6", "#0F6B5C", "#B5502E", "#C99A2E", "#5B6478", "#E3D8C3"
FONT = "Inter, Arial, sans-serif"

def svg(w, h, body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">{body}</svg>'

def t(x, y, s, size=14, fill=INK, anchor="start", weight=400, extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" {extra}>{s}</text>'

# ---------------------------------------------------------------- compute chart
def compute_chart():
    # Training compute (FLOP) of landmark models — public estimates (Epoch AI / papers), order of magnitude
    pts = [(1998, 2.8e12, "LeNet-5"), (2012, 4.7e17, "AlexNet"), (2014, 1.2e19, "VGG-16"), (2015, 1.2e19, "ResNet-152"),
           (2017, 2.3e19, "Transformer"), (2018, 2.9e20, "BERT-L"), (2020, 3.1e23, "GPT-3"), (2022, 2.5e24, "PaLM"),
           (2023, 2.1e25, "GPT-4*"), (2024, 3.8e25, "Llama 3.1-405B")]
    W, H, L, R, T, B = 760, 470, 70, 20, 20, 50
    x0, x1, y0, y1 = 1996, 2029, 12, 26
    X = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    Y = lambda v: H - B - (math.log10(v) - y0) / (y1 - y0) * (H - T - B)
    b = f'<rect width="{W}" height="{H}" fill="none"/>'
    for e in range(y0, y1 + 1, 2):
        b += f'<line x1="{L}" x2="{W-R}" y1="{Y(10**e):.1f}" y2="{Y(10**e):.1f}" stroke="{LINE}" stroke-width="1"/>'
        b += t(L - 10, Y(10**e) + 5, f"10<tspan dy='-7' font-size='11'>{e}</tspan>", 14, MUT, "end")
    for yr in range(2000, 2027, 5):
        b += t(X(yr), H - B + 24, str(yr), 14, MUT, "middle")
    # deep-learning era shading
    b += f'<rect x="{X(2012):.1f}" y="{T}" width="{X(2029)-X(2012):.1f}" height="{H-T-B}" fill="{SAF}" opacity="0.08"/>'
    b += t(X(2012) + 8, T + 18, "deep-learning era (GPUs)", 13, "#8A6512", "start", 600)
    for i, (yr, f, n) in enumerate(pts):
        c = TER if yr >= 2020 else (MAJ if yr >= 2012 else ZEL)
        b += f'<circle cx="{X(yr):.1f}" cy="{Y(f):.1f}" r="7" fill="{c}" stroke="#fff" stroke-width="2"/>'
        off = {"Llama 3.1-405B": (12, -4, "start"), "GPT-4*": (-4, -14, "end"), "PaLM": (12, 16, "start"),
               "ResNet-152": (-12, -8, "end"), "VGG-16": (10, 20, "start"), "Transformer": (10, 14, "start"),
               "BERT-L": (10, -8, "start"), "GPT-3": (10, 14, "start"), "AlexNet": (10, 14, "start")}
        dx, dy, anc = off.get(n, (10, -10, "start"))
        b += t(X(yr) + dx, Y(f) + dy, n, 14, INK, anc, 600)
    b += t(14, T + (H - T - B) / 2, "training compute (FLOP, log)", 13, MUT, "middle", 500, f'transform="rotate(-90 14 {T + (H-T-B)/2})"')
    open(D + "chart_compute.svg", "w").write(svg(W, H, b))

# ---------------------------------------------------------------- chips chart
def chips_chart():
    chips = [("GTX 580\n2010", 1.58, "FP32"), ("V100\n2017", 125, "FP16 tensor"), ("A100\n2020", 312, "FP16/BF16"),
             ("H100\n2022", 989, "FP16/BF16"), ("B200\n2024", 2250, "FP16/BF16")]
    W, H, L, R, T, B = 640, 470, 64, 20, 30, 70
    y0, y1 = 0, 4
    Y = lambda v: H - B - (math.log10(v) - y0) / (y1 - y0) * (H - T - B)
    bw = (W - L - R) / len(chips)
    b = ""
    for e in range(y0, y1 + 1):
        b += f'<line x1="{L}" x2="{W-R}" y1="{Y(10**e):.1f}" y2="{Y(10**e):.1f}" stroke="{LINE}"/>'
        b += t(L - 10, Y(10**e) + 5, f"{10**e:g}", 14, MUT, "end")
    cols = [ZEL, MAJ, MAJ, TER, TER]
    for i, (n, v, p) in enumerate(chips):
        x = L + i * bw + bw * 0.18
        b += f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw*0.64:.1f}" height="{H-B-Y(v):.1f}" rx="8" fill="{cols[i]}"/>'
        b += t(x + bw * 0.32, Y(v) - 10, f"{v:g}", 16, INK, "middle", 700)
        a, c = n.split("\n")
        b += t(x + bw * 0.32, H - B + 24, a, 15, INK, "middle", 600)
        b += t(x + bw * 0.32, H - B + 44, c + " · " + p, 12, MUT, "middle")
    b += t(14, T + (H - T - B) / 2, "peak dense TFLOPS (log)", 13, MUT, "middle", 500, f'transform="rotate(-90 14 {T + (H-T-B)/2})"')
    open(D + "chart_chips.svg", "w").write(svg(W, H, b))

# ---------------------------------------------------------------- activations
def activations():
    fs = [("Sigmoid", lambda x: 1 / (1 + math.exp(-x)), (-0.2, 1.2)),
          ("tanh", math.tanh, (-1.2, 1.2)),
          ("ReLU", lambda x: max(0, x), (-0.5, 4)),
          ("GELU", lambda x: 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))), (-0.5, 4))]
    out = []
    for name, f, (lo, hi) in fs:
        W, H = 300, 190
        X = lambda v: 20 + (v + 4) / 8 * (W - 40)
        Y = lambda v: H - 20 - (v - lo) / (hi - lo) * (H - 40)
        b = f'<line x1="20" x2="{W-20}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="{LINE}" stroke-width="2"/>'
        b += f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="15" y2="{H-15}" stroke="{LINE}" stroke-width="2"/>'
        d = "M" + " L".join(f"{X(-4 + 8*i/200):.1f},{Y(f(-4 + 8*i/200)):.1f}" for i in range(201))
        b += f'<path d="{d}" fill="none" stroke="{TER}" stroke-width="4" stroke-linecap="round"/>'
        b += t(28, 30, name, 20, INK, "start", 700)
        open(D + f"act_{name.lower()}.svg", "w").write(svg(W, H, b))

# ---------------------------------------------------------------- nested arches AI ⊃ ML ⊃ DL ⊃ GenAI
def nested_arches():
    W, H = 760, 560
    layers = [("Artificial Intelligence", "1956 →  reasoning, search, rules, learning", INK, 720),
              ("Machine Learning", "1959 →  learn from data", MAJ, 580),
              ("Deep Learning", "2006/2012 →  learned representations", ZEL, 440),
              ("Generative AI &amp; Foundation Models", "2017 →  transformers, LLMs, GeoFMs", TER, 300)]
    b = ""
    base = H - 10
    for i, (n, d, c, w) in enumerate(layers):
        x = (W - w) / 2
        h = 520 - i * 120
        r = w / 2
        top = base - h
        path = f"M{x},{base} L{x},{top + r*0.9:.1f} A{r},{r*0.9:.1f} 0 0 1 {x+w},{top + r*0.9:.1f} L{x+w},{base} Z"
        b += f'<path d="{path}" fill="{c}" stroke="{SAF}" stroke-width="3"/>'
        b += t(W / 2, top + 44, n, 24 if i < 3 else 19, "#fff", "middle", 700)
        b += t(W / 2, top + 70, d, 14, "#E6E9F2", "middle", 400)
    open(D + "nested_arches.svg", "w").write(svg(W, H, b))

if __name__ == "__main__":
    compute_chart(); chips_chart(); activations(); nested_arches()
    print("charts ok")
