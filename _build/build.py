#!/usr/bin/env python3
"""Assemble the course website from slide partials.

src/slides/*.html  -> site/index.html
 - <pre class="py|js|bash|yaml" data-file=".." data-size="sm|xs">code</pre>  -> highlighted code card
 - $$ ... $$  (display)  and  \\( ... \\)  (inline)  -> KaTeX HTML (rendered at build time, no JS needed)
 - <section data-title data-module data-kind data-src> + <h2> -> header, footer, slide number
"""
import re, json, subprocess, html, glob, os, sys
from pygments import highlight
from pygments.lexers import PythonLexer, JavascriptLexer, BashLexer, YamlLexer, JsonLexer
from pygments.formatters import HtmlFormatter

ROOT = os.path.dirname(os.path.abspath(__file__))
SITE = os.environ.get("SITE_DIR", os.path.dirname(ROOT))  # the website root (parent of _build)
LEX = {"py": PythonLexer, "js": JavascriptLexer, "bash": BashLexer, "yaml": YamlLexer, "json": JsonLexer}
KIND_LABEL = {"theory": "Theory", "practice": "Practice", "lab": "Lab", "case": "Case study"}

def code_block(m):
    attrs, body = m.group(1), m.group(2)
    lang = re.search(r'class="(\w+)"', attrs).group(1)
    fname = html.unescape((re.search(r'data-file="([^"]*)"', attrs) or [None, lang])[1])
    size = (re.search(r'data-size="([^"]*)"', attrs) or [None, ""])[1]
    style = (re.search(r'style="([^"]*)"', attrs) or [None, ""])[1]
    body = html.unescape(body).strip("\n")
    body = re.sub(r"^\n+", "", body)
    # dedent
    lines = body.split("\n")
    ind = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    body = "\n".join(l[ind:] for l in lines)
    hl = highlight(body, LEX[lang](), HtmlFormatter(nowrap=True))
    return (f'<div class="code {size}" style="{style}"><div class="bar"><span class="dots"><i></i><i></i><i></i></span>'
            f'<span class="fname">{html.escape(fname)}</span><button class="copy">Copy</button></div>'
            f'<pre>{hl}</pre></div>')

def build():
    parts = sorted(glob.glob(os.path.join(ROOT, "src/slides/*.html")))
    raw = "\n".join(open(p, encoding="utf-8").read() for p in parts)

    # 1) protect code
    codes = []
    def stash(m):
        codes.append(code_block(m)); return f"@@CODE{len(codes)-1}@@"
    raw = re.sub(r"<pre ([^>]*class=\"(?:py|js|bash|yaml|json)\"[^>]*)>(.*?)</pre>", stash, raw, flags=re.S)

    # 2) math
    maths = []
    def mstash(m, display):
        maths.append({"tex": html.unescape(m.group(1)), "display": display}); return f"@@MATH{len(maths)-1}@@"
    raw = re.sub(r"\$\$(.+?)\$\$", lambda m: mstash(m, True), raw, flags=re.S)
    raw = re.sub(r"\\\((.+?)\\\)", lambda m: mstash(m, False), raw, flags=re.S)
    json.dump(maths, open(os.path.join(ROOT, ".maths.json"), "w"))
    subprocess.run(["node", os.path.join(ROOT, "katex_render.js"), os.path.join(ROOT, ".maths.json"), os.path.join(ROOT, ".maths_out.json")], check=True)
    rendered = json.load(open(os.path.join(ROOT, ".maths_out.json")))
    raw = re.sub(r"@@MATH(\d+)@@", lambda m: rendered[int(m.group(1))], raw)
    raw = re.sub(r"@@CODE(\d+)@@", lambda m: codes[int(m.group(1))], raw)

    # 3) slides: header / footer / numbering
    secs = re.findall(r"<section((?:\"[^\"]*\"|[^>\"])*)>(.*?)</section>", raw, flags=re.S)
    out = []
    for i, (attrs, body) in enumerate(secs, 1):
        a = dict(re.findall(r'(data-[\w-]+|class)="([^"]*)"', attrs))
        cls = "slide " + a.get("class", "")
        kind = a.get("data-kind", "theory")
        module = a.get("data-module", "")
        src = a.get("data-src", "")
        title = a.get("data-title", "")
        dark = "dark" in cls
        if not dark:
            h2 = re.search(r"<h2>(.*?)</h2>", body, flags=re.S)
            if h2:
                tag = f'<span class="tag {kind}">{KIND_LABEL.get(kind, kind)}</span>'
                head = f'<header class="s-head"><div class="kicker"><span>{module}</span>{tag}</div><h2>{h2.group(1)}</h2></header>'
                body = body.replace(h2.group(0), head, 1)
                if not title:
                    title = re.sub("<.*?>", "", h2.group(1))
        foot = f'<footer class="s-foot"><span class="src">{src}</span><span class="num">{i}</span></footer>'
        body = body.rstrip() + "\n" + foot
        title_attr = html.escape(re.sub("<.*?>", "", title), quote=True)
        out.append(f'<section class="{cls.strip()}" data-title="{title_attr}" data-module="{html.escape(module, quote=True)}" data-kind="{kind}">{body}\n</section>')

    tpl = open(os.path.join(ROOT, "src/template.html"), encoding="utf-8").read()
    page = tpl.replace("<!--SLIDES-->", "\n".join(out)).replace("{{N}}", str(len(out)))
    open(os.path.join(SITE, "index.html"), "w", encoding="utf-8").write(page)
    print(f"built {len(out)} slides, {len(maths)} equations, {len(codes)} code blocks")

if __name__ == "__main__":
    build()
