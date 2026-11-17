#!/usr/bin/env python3
"""Assemble the course website from slide partials.

src/slides/*.html  -> site/index.html
 - <pre class="py|js|bash|yaml" data-file=".." data-size="sm|xs">code</pre>  -> highlighted code card
 - $$ ... $$  (display)  and  \\( ... \\)  (inline)  -> KaTeX HTML (rendered at build time, no JS needed)
 - <section data-title data-module data-kind data-src> + <h2> -> header, footer, slide number

Settings (settings.toml, or the file named by $COURSE_SETTINGS):
 - any element with data-lang="ar" is dropped when arabic = false
 - any element with data-require="flag" is dropped when [modules].flag is false
   ("!flag" keeps the element only when the flag is false)
 - sections with data-part="I" (optionally data-extra="flag") drive the schedule; the
   placeholders {{slot:I}}, {{start:I}}, {{total_min}}, {{hours}}, {{duration}} and
   {{instructor}} are filled in from it.
"""
import re, json, subprocess, html, glob, os, sys
from pygments import highlight
from pygments.lexers import PythonLexer, JavascriptLexer, BashLexer, YamlLexer, JsonLexer
from pygments.formatters import HtmlFormatter
try:
    import tomllib                      # Python 3.11+
except ModuleNotFoundError:             # pragma: no cover
    import tomli as tomllib             # pip install tomli

ROOT = os.path.dirname(os.path.abspath(__file__))
SITE = os.environ.get("SITE_DIR", os.path.dirname(ROOT))  # the website root (parent of _build)
LEX = {"py": PythonLexer, "js": JavascriptLexer, "bash": BashLexer, "yaml": YamlLexer, "json": JsonLexer}
KIND_LABEL = {"theory": "Theory", "practice": "Practice", "lab": "Lab", "case": "Case study"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


# ------------------------------------------------------------------ settings
def load_settings():
    path = os.environ.get("COURSE_SETTINGS", os.path.join(ROOT, "settings.toml"))
    with open(path, "rb") as f:
        s = tomllib.load(f)
    s.setdefault("arabic", True)
    s.setdefault("instructor", "")
    s.setdefault("modules", {})
    s.setdefault("schedule", {})
    s.setdefault("hide_pages", [])
    for k, v in s["modules"].items():
        if not isinstance(v, bool):
            sys.exit(f"settings: [modules].{k} must be true or false, got {v!r}")
    return s


def flag_on(settings, name):
    neg = name.startswith("!")
    name = name.lstrip("!")
    if name not in settings["modules"]:
        sys.exit(f"settings: data-require=\"{name}\" is used in the slides but [modules].{name} is not set")
    return settings["modules"][name] != neg


# ------------------------------------------------------- optional elements
TAG = re.compile(r"<([a-zA-Z][\w-]*)((?:\"[^\"]*\"|'[^']*'|[^>\"'])*)>")


def drop_elements(src, keep):
    """Remove every element whose opening-tag attributes fail keep(attrs), with its whole subtree."""
    out, pos = [], 0
    while True:
        m = TAG.search(src, pos)
        if not m:
            break
        tag, attrs = m.group(1).lower(), m.group(2)
        if keep(attrs):
            out.append(src[pos:m.end()])
            pos = m.end()
            continue
        out.append(src[pos:m.start()])
        if tag in VOID or attrs.rstrip().endswith("/"):
            pos = m.end()
            continue
        depth, j = 1, m.end()
        tagpat = re.compile(rf"<(/?){tag}\b(?:\"[^\"]*\"|'[^']*'|[^>\"'])*>", re.I)
        while depth:
            n = tagpat.search(src, j)
            if not n:
                sys.exit(f"build: unclosed <{tag}> with {attrs.strip()[:80]}")
            depth += -1 if n.group(1) else 1
            j = n.end()
        pos = j
    out.append(src[pos:])
    return "".join(out)


def make_keep(settings):
    def keep(attrs):
        if 'data-lang="ar"' in attrs and not settings["arabic"]:
            return False
        req = re.search(r'data-require="([^"]+)"', attrs)
        return not req or all(flag_on(settings, f.strip()) for f in req.group(1).split(","))
    return keep


# ---------------------------------------------------------------- hide_pages
def hidden_slides(secs, spec):
    """1-based indices of the slides to hide. spec: list of ints / "3-7" ranges / exact slide titles,
    or one string such as "1,2,14"."""
    if isinstance(spec, (int, str)):
        spec = [spec]
    items = []
    for it in spec:
        if isinstance(it, int):
            items.append(it)
        elif re.fullmatch(r"[\d\s,\-–]+", str(it)):
            for part in str(it).split(","):
                part = part.strip()
                m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
                if m:
                    items += range(int(m.group(1)), int(m.group(2)) + 1)
                elif part:
                    items.append(int(part))
        else:
            items.append(str(it).strip())
    titles = []
    for attrs, body in secs:
        t = re.search(r'data-title="([^"]*)"', attrs)
        h2 = re.search(r"<h2>(.*?)</h2>", body, flags=re.S)
        titles.append(html.unescape(t.group(1) if t else re.sub("<.*?>", "", h2.group(1)) if h2 else ""))
    hide = set()
    for it in items:
        if isinstance(it, int):
            if 1 <= it <= len(secs):
                hide.add(it)
            else:
                print(f"settings: hide_pages: there is no slide {it} (the deck has {len(secs)})", file=sys.stderr)
        else:
            hits = [i for i, t in enumerate(titles, 1) if t == it]
            if not hits:
                print(f"settings: hide_pages: no slide titled {it!r}", file=sys.stderr)
            hide.update(hits)
    return hide


# ------------------------------------------------------------------ schedule
def hm(minutes):
    return f"{minutes // 60}:{minutes % 60:02d}"


def hours_label(total):
    h, m = divmod(total, 60)
    frac = {0: "", 15: "¼", 30: "½", 45: "¾"}
    return f"{h}{frac[m]}" if m in frac else f"{total / 60:.1f}"


def schedule(raw, settings):
    sch, extra_min = settings["schedule"], settings["schedule"].get("modules", {})
    vals, t = {}, 0
    for attrs in re.findall(r"<section((?:\"[^\"]*\"|[^>\"])*)>", raw):
        part = re.search(r'data-part="([^"]+)"', attrs)
        if not part:
            continue
        p = part.group(1)
        if p not in sch:
            sys.exit(f"settings: [schedule] has no minutes for part {p!r}")
        mins = int(sch[p])
        ex = re.search(r'data-extra="([^"]+)"', attrs)
        for f in (ex.group(1).split(",") if ex else []):
            f = f.strip()
            if settings["modules"].get(f):
                mins += int(extra_min.get(f, 0))
        vals[f"start:{p}"] = hm(t)
        vals[f"min:{p}"] = str(mins)
        vals[f"slot:{p}"] = f"{hm(t)} – {hm(t + mins)} · {mins} min"
        t += mins
    h = hours_label(t)
    vals.update({"total_min": str(t), "hours": f"{h} hours", "duration": f"{h}-hour",
                 "instructor": html.escape(settings["instructor"])})
    return vals


def fill(text, vals):
    def rep(m):
        key = m.group(1)
        if key not in vals:
            sys.exit(f"build: unknown placeholder {{{{{key}}}}}")
        return vals[key]
    return re.sub(r"\{\{([a-z_]+(?::[\w]+)?|N)\}\}", rep, text)


# ---------------------------------------------------------------- code cards
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
    settings = load_settings()
    keep = make_keep(settings)
    parts = sorted(glob.glob(os.path.join(ROOT, "src/slides/*.html")))
    raw = "\n".join(open(p, encoding="utf-8").read() for p in parts)

    # 0) settings: optional elements, then schedule + placeholders
    raw = drop_elements(raw, keep)
    vals = schedule(raw, settings)

    # 1) protect code
    codes = []
    def stash(m):
        codes.append(code_block(m)); return f"@@CODE{len(codes)-1}@@"
    raw = re.sub(r"<pre ([^>]*class=\"(?:py|js|bash|yaml|json)\"[^>]*)>(.*?)</pre>", stash, raw, flags=re.S)
    raw = fill(raw, vals)

    # 2) math
    maths = []
    def mstash(m, display):
        maths.append({"tex": html.unescape(m.group(1)), "display": display}); return f"@@MATH{len(maths)-1}@@"
    raw = re.sub(r"\$\$(.+?)\$\$", lambda m: mstash(m, True), raw, flags=re.S)
    raw = re.sub(r"\\\((.+?)\\\)", lambda m: mstash(m, False), raw, flags=re.S)
    mj, mo = os.path.join(ROOT, ".maths.json"), os.path.join(ROOT, ".maths_out.json")
    with open(mj, "w", encoding="utf-8") as f:
        json.dump(maths, f)
    subprocess.run(["node", os.path.join(ROOT, "katex_render.js"), mj, mo], check=True)
    with open(mo, encoding="utf-8") as f:
        rendered = json.load(f)
    raw = re.sub(r"@@MATH(\d+)@@", lambda m: rendered[int(m.group(1))], raw)
    raw = re.sub(r"@@CODE(\d+)@@", lambda m: codes[int(m.group(1))], raw)

    # 3) slides: header / footer / numbering
    secs = re.findall(r"<section((?:\"[^\"]*\"|[^>\"])*)>(.*?)</section>", raw, flags=re.S)
    hide = hidden_slides(secs, settings["hide_pages"])
    secs = [x for i, x in enumerate(secs, 1) if i not in hide]
    out, demos = [], 0
    for i, (attrs, body) in enumerate(secs, 1):
        a = dict(re.findall(r'(data-[\w-]+|class)="([^"]*)"', attrs))
        cls = "slide " + a.get("class", "")
        kind = a.get("data-kind", "theory")
        module = a.get("data-module", "")
        src = a.get("data-src", "")
        title = a.get("data-title", "")
        dark = "dark" in cls
        demos += len(re.findall(r'data-demo="', body))
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
    tpl = drop_elements(tpl, keep)
    vals["N"] = str(len(out))
    page = fill(tpl, vals).replace("<!--SLIDES-->", "\n".join(out))
    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    on = [k for k, v in settings["modules"].items() if v]
    print(f"built {len(out)} slides, {len(maths)} equations, {len(codes)} code blocks, {demos} interactive demos · "
          f"{vals['hours']} ({vals['total_min']} min) · arabic={'on' if settings['arabic'] else 'off'} · "
          f"modules on: {', '.join(on) or 'none'}" + (f" · hidden: {', '.join(map(str, sorted(hide)))}" if hide else ""))


if __name__ == "__main__":
    build()
