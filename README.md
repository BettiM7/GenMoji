# GenMoji

University Generative AI project for text-to-emoji generation.

```text
GenMoji/
├── shared/                  # Team prompts, configuration and evaluation protocol
├── sana-emoji/              # Person 2: Sana flow-matching baseline
│   ├── run_sana_baseline.py
│   ├── test_sana.py          # Original working test
│   ├── requirements-baseline.txt
│   ├── README.md            # Setup, execution and validation details
│   └── results/             # Generated images and metadata (ignored by Git)
└── work_split.md            # Team responsibilities
```

Keep model-specific scripts, dependencies and results inside their contribution folder. Keep common evaluation inputs in `shared/`.

See [Sana setup and usage](sana-emoji/README.md).

```bash
cd sana-emoji
source .venv/bin/activate
python run_sana_baseline.py --limit 3 --seeds 42 --output-dir results/smoke_seed42 --local-files-only
```
