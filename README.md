# GenMoji

Text-to-emoji project using **Stable Diffusion 1.5 + LoRA**, **Sana 0.6B (512px)**,
and compositional guidance. Responsibilities and research scope are in
[work_split.md](work_split.md).

## Setup and prepare the shared dataset

Use **Python 3.12 in a local `.venv` virtual environment** and install with pip.
Data preparation runs on CPU; downloading model weights is unnecessary here.

Run these commands from the repository root.

Create and activate the Python 3.12 environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data]'
python --version
python -c "import sys; print(sys.executable)"
```

The version should be `3.12.x` and the executable should be inside this project's
`.venv/bin/`. Activate the environment with `source .venv/bin/activate` in each new
terminal. Select `.venv/bin/python` as the interpreter in your IDE.

CairoSVG is the Python library that renders OpenMoji's vector SVG artwork into
the 512 × 512 PNG images used by both models. Rendering directly from SVG keeps
edges sharp at the target resolution. It is only needed for data preparation.
CairoSVG also needs the native Cairo library. If importing it reports that Cairo
cannot be found, install it with `brew install cairo` on macOS, or
`sudo apt-get install libcairo2` on Ubuntu. If Homebrew's library is not detected,
run preparation with `DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix cairo)/lib"`
set for that command.

Download a fixed OpenMoji release, with only color SVGs and metadata:

```bash
git clone --depth 1 --filter=blob:none --sparse --branch 17.0.0 \
  https://github.com/hfg-gmuend/openmoji.git data/raw/openmoji
git -C data/raw/openmoji sparse-checkout set color/svg data
genmoji prepare
```

Root files, including `LICENSE.txt`, are retained by Git's cone-mode sparse checkout.
Full preparation uses all entries in the release metadata. For a quick conversion
check first, use a separate output directory:

```bash
genmoji prepare --limit 20 --output data/processed/smoke-512
```

See [the dataset guide](docs/dataset.md) for splits, captions, licensing,
validation, and exact ingestion commands for both models.

## Project structure

```text
src/genmoji/
  cli.py                  # genmoji prepare CLI
  data.py                 # SVG rasterization, captions, family splits, provenance
data/
  raw/openmoji/           # downloaded source; ignored by Git
  processed/openmoji-512/ # generated data; ignored by Git
    imagefolder/
      train/              # RGB PNGs + metadata.jsonl
      validation/         # same format
      test/               # same format
    provenance/           # source hashes, authors, settings, license
experiments/
  diffusion/README.md     # Person 1: SD baseline and LoRA integration
  flow_matching/README.md # Person 2: Sana integration and feasibility
  guidance/README.md      # Person 3: shared SD data and prompt conventions
scripts/
  check_dataset.py        # verify ImageFolder decoding and split isolation
docs/dataset.md           # data contract and training entry points
outputs/                  # model adapters, images, logs; ignored by Git
tests/                    # data pipeline regression tests
```

The experiment folders are implementation handoffs; model training and custom
guidance are not implemented yet. All methods should share the prepared data,
prompt set, seeds, and 512px output size.

```bash
python scripts/check_dataset.py data/processed/openmoji-512
python -m unittest discover -s tests -v
```

OpenMoji artwork is by [OpenMoji](https://openmoji.org), licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Source attribution and the original license are included with every prepared dataset.
