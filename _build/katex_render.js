let katex;
try { katex = require("katex"); } catch (e) { katex = require("/opt/npm-tools/node_modules/katex"); }
const fs = require("fs");
const items = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = items.map(({ tex, display }) => {
  try {
    const h = katex.renderToString(tex, { displayMode: display, throwOnError: true, output: "html", strict: false });
    return h;
  } catch (e) {
    console.error("KaTeX error:", e.message, "\n  in:", tex);
    process.exitCode = 1;
    return `<code>${tex}</code>`;
  }
});
fs.writeFileSync(process.argv[3], JSON.stringify(out));
