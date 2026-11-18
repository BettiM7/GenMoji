"""Download eight pinned OpenMoji assets, retaining attribution and source hashes."""
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = Path(__file__).resolve().parent
DATA = BASE / "data/openmoji"
TAG = "17.0.0"
PAIRS = [("1F34E", "A red apple"), ("1F34C", "A yellow banana"),
         ("1F697", "A red car"), ("1F680", "A space rocket"),
         ("26BD", "A soccer ball"), ("1F3B8", "A guitar"),
         ("1F3E0", "A house"), ("1F355", "A slice of pizza")]

if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    records = []
    for code, caption in PAIRS:
        url = f"https://raw.githubusercontent.com/hfg-gmuend/openmoji/{TAG}/color/618x618/{code}.png"
        path = DATA / f"{code}.png"
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(url).read())
        records.append({"file": path.name, "caption": caption + ", OpenMoji emoji illustration on a white background",
                        "source": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "license": "CC-BY-SA-4.0", "modifications": "None to source PNG; runtime composites transparency on white and resizes to 512x512"})
    (DATA / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    (DATA / "LICENSE.txt").write_bytes(urllib.request.urlopen(
        f"https://raw.githubusercontent.com/hfg-gmuend/openmoji/{TAG}/LICENSE.txt").read())
    (DATA / "ATTRIBUTION.md").write_text(
        "# OpenMoji feasibility subset\n\nEmojis designed by [OpenMoji](https://openmoji.org/), "
        "the open-source emoji and icon project by HfG Schwäbisch Gmünd and contributors.\n\n"
        "Source release: 17.0.0. License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). "
        "See LICENSE.txt and manifest.json for per-file sources and hashes.\n\n"
        "Captions are project-authored. No evaluation prompts or generated baseline images are used. "
        "Only the first pair is used for the single optimizer update; all eight images are loaded and validated.\n")
    print(f"Prepared {len(records)} licensed image-caption pairs at {DATA}")
