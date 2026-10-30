"""Ubah train.py (penanda '# %%') menjadi analisis_lstm.ipynb."""
import json, re
from pathlib import Path

src = Path("train.py").read_text(encoding="utf-8")
cells = []
for block in re.split(r"(?m)^# %%", src)[1:]:
    head, _, body = block.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in head:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body.splitlines(True)})
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}}
Path("analisis_lstm.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("analisis_lstm.ipynb:", len(cells), "sel")
