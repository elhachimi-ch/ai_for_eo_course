import json, os, pathlib
from playwright.sync_api import sync_playwright
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "export"); SITE = os.environ.get("SITE_DIR", os.path.dirname(HERE))
os.makedirs(OUT + "/img", exist_ok=True)
with sync_playwright() as p:
    # EXPORT_BROWSER=msedge|chrome uses an installed browser instead of Playwright's bundled Chromium
    exe = "/usr/bin/google-chrome" if os.path.exists("/usr/bin/google-chrome") else None
    channel = os.environ.get("EXPORT_BROWSER")
    b = p.chromium.launch(channel=channel, args=["--no-sandbox"]) if channel else p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1.2)
    pg.goto(pathlib.Path(SITE, "index.html").resolve().as_uri() + "?export"); pg.wait_for_function("window.__deckReady===true")
    pg.evaluate("document.fonts.ready"); pg.wait_for_timeout(1200)
    meta = pg.evaluate("""() => [...document.querySelectorAll('.slide')].map(s => ({
        title: s.dataset.title, module: s.dataset.module,
        notes: (s.querySelector('.notes')||{innerText:''}).innerText.trim(),
        src: (s.querySelector('.s-foot .src')||{innerText:''}).innerText.trim()}))""")
    deck = pg.evaluate("""() => { const m = (n) => (document.querySelector(`meta[name="${n}"]`)||{}).content || "";
        return {author: m('author'), duration: m('course-duration'), title: m('course-title')}; }""")
    for i, s in enumerate(pg.query_selector_all(".slide"), 1):
        png = f"{OUT}/img/s{i:03d}.png"
        s.screenshot(path=png)
        Image.open(png).convert("RGB").save(png.replace(".png", ".jpg"), quality=88, optimize=True)
        os.remove(png)
    json.dump(meta, open(f"{OUT}/meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(deck, open(f"{OUT}/deck.json", "w", encoding="utf-8"), ensure_ascii=False)
    pg.emulate_media(media="print")
    pg.pdf(path=f"{SITE}/downloads/DL4EO_course.pdf", width="1600px", height="900px", print_background=True,
           margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}, prefer_css_page_size=True)
    b.close()
print(len(meta), "slides exported")
