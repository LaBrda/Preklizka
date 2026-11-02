"""Překlad zadání pro AI kreslíře z češtiny do angličtiny (Helsinki-NLP opus-mt-cs-en, Apache 2.0).

  python translate.py vstup.txt    -> na výstup řádek PREKLAD {"en": "..."}
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # vložený Python nepřidává složku skriptu
from repaint import model_dir  # noqa: E402  (přidá i složku knihoven kreslíře)

if __name__ == "__main__":
    from transformers import MarianMTModel, MarianTokenizer
    with open(sys.argv[1], encoding="utf-8") as f:
        text = f.read().strip()
    d = os.path.join(model_dir(), "mt")
    td = d
    if not d.isascii():                                   # sentencepiece ve Windows neotevře cestu s diakritikou (Překližka…)
        import shutil
        import tempfile
        td = os.path.join(tempfile.gettempdir(), "preklizka_mt")
        if not td.isascii():
            td = os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "ProgramData", "preklizka_mt")
        os.makedirs(td, exist_ok=True)
        for f in ("source.spm", "target.spm", "vocab.json", "tokenizer_config.json", "config.json"):
            if os.path.exists(os.path.join(d, f)):
                shutil.copyfile(os.path.join(d, f), os.path.join(td, f))
    tok = MarianTokenizer.from_pretrained(td)
    model = MarianMTModel.from_pretrained(d)
    parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]   # po částech: krátké fráze se překládají věrněji
    out = []
    for p in parts:
        ids = tok([p], return_tensors="pt", truncation=True)
        out.append(tok.decode(model.generate(**ids, max_new_tokens=60, num_beams=4)[0], skip_special_tokens=True).strip(" ."))
    print("PREKLAD " + json.dumps({"en": ", ".join(out)}, ensure_ascii=False), flush=True)
