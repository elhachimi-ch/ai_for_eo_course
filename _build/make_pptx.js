const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const meta = JSON.parse(fs.readFileSync(path.join(__dirname, "export", "meta.json"), "utf8"));
// author and duration come from settings.toml via the built page (written by export.py)
const deckFile = path.join(__dirname, "export", "deck.json");
const deck = fs.existsSync(deckFile) ? JSON.parse(fs.readFileSync(deckFile, "utf8")) : {};
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";             // 13.333 x 7.5 in (16:9)
pres.author = deck.author || "";
pres.title = "Deep Learning for Earth Observation";
pres.subject = `${deck.duration || "3-hour"} course: from classical machine learning and perceptrons to foundation models and agents`;
let lastModule = null;
meta.forEach((m, i) => {
  const mod = (m.module || "Course").replace(/&amp;/g, "&");
  if (mod !== lastModule) { pres.addSection({ title: mod }); lastModule = mod; }
  const s = pres.addSlide({ sectionTitle: mod });
  const n = String(i + 1).padStart(3, "0");
  s.addImage({ path: path.join(__dirname, "export", "img", `s${n}.jpg`), x: 0, y: 0, w: 13.333, h: 7.5, altText: m.title || `Slide ${i + 1}` });
  const notes = [m.title, m.notes, m.src ? "Sources: " + m.src : ""].filter(Boolean).join("\n\n");
  s.addNotes(notes);
});
pres.writeFile({ fileName: path.join(__dirname, "..", "downloads", "DL4EO_course.pptx") }).then(f => console.log("wrote", f));
