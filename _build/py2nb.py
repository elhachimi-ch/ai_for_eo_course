"""Convert '# %%' percent-format .py files into Jupyter notebooks (nbformat v4, no dependency)."""
import json, sys, os, re, glob

def convert(src, dst):
    text = open(src, encoding="utf-8").read()
    blocks = re.split(r"^# %%(.*)$", text, flags=re.M)
    cells = []
    # blocks: [preamble, tag1, body1, tag2, body2, ...]
    for i in range(1, len(blocks), 2):
        tag, body = blocks[i].strip(), blocks[i + 1].strip("\n")
        if "[markdown]" in tag:
            lines = [re.sub(r"^# ?", "", l) for l in body.split("\n")]
            src_lines = "\n".join(lines).strip("\n")
            cells.append({"cell_type": "markdown", "metadata": {}, "source": src_lines.splitlines(True)})
        else:
            if not body.strip():
                continue
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": body.rstrip().splitlines(True)})
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU",
                       "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(dst, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

if __name__ == "__main__":
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.py"))):
        name = os.path.splitext(os.path.basename(f))[0]
        convert(f, os.path.join(out, name + ".ipynb"))
        print("ok", name)
