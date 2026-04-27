# SIOP 2026 ML Competition — One Hot Key

Automated extraction of zero-order bivariate Pearson r values from academic PDFs in industrial-organizational psychology, built as a one-person-plus-agents experiment for the SIOP 2026 ML Competition.

## Headline results

| Phase    | Studies | MSE      | Rank   | Submission             |
|----------|---------|----------|--------|------------------------|
| Dev set  | 127     | 0.013641 | 6 / 10 | submission_v11_study59 |
| Test set | 66      | 0.0351   | TBD    | 2026-04-11             |

Dev-set rank places this submission within 0.0005 MSE of the lab-team and research-squad submissions ranked above it.

## The experiment

> This pipeline was built and run by a single developer (Matt) operating with two AI agents — Claude (diagnostics, PDF reading, prompt-writing, file validation) and Cursor (all code implementation). PowerShell drove batch runs and spot checks; no iterative debugging happened in shell.
>
> The experiment is implicit in the leaderboard: a one-person-plus-agents team competing against teams of researchers, analysts, and grad students. The architecture and division of labor matter as much as the extraction accuracy. See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the technical design and [the SIOP 2026 deck](docs/one_hot_key_siop_2026.pdf) for the presentation.

## Architecture summary

> Each PDF passes through a 4-tier extraction cascade: Tier 0 (pdfplumber geometric tables), Tier 1 (Docling ML TableFormer with optional qwen2.5-VL crop validation), Tier 1b (qwen2.5-VL on rendered page images), and Tier 2 (regex candidates classified by phi4). PyMuPDF (fitz) sits underneath every tier as the page reader, rasterizer, and region cropper. A high-confidence value at any tier short-circuits the rest. See [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repo structure

    siop-2026-pipelines/
    ├── README.md                        ← you are here
    ├── CHANGELOG.md                     v9 → v10 → v11 → v12 evolution
    ├── pipeline_dev.py                  Dev pipeline (127 trust × wellbeing)
    ├── pipeline_test.py                 Test pipeline (66 papers, 23 pairs)
    ├── geom_corr_matrix.py              Geometric matrix helper (imported by both)
    ├── opus_sweep_v10.py                Targeted Opus extraction (separate module)
    ├── data/
    │   ├── dev_articles.csv
    │   ├── test_articles.csv
    │   ├── test_construct_definitions.csv
    │   └── ConstructDefinitions.txt
    ├── docs/
    │   ├── ARCHITECTURE.md              4-tier cascade design rationale
    │   ├── KNOWN_LANDMINES.md           Silent-failure traps + guards
    │   ├── META_ANALYSIS_MANUAL.md      Adaptation guide
    │   └── one_hot_key_siop_2026.pdf    Presentation deck (optional)
    └── pdfs/                            Source PDFs (gitignored; see data/README.md)

geom_corr_matrix.py may live in the parent 2026-meta-open/ folder; add it to PYTHONPATH if imports fail from this directory alone.

## Quick start

```bash
pip install pymupdf pdfplumber docling ollama scipy numpy
ollama pull phi4
ollama pull qwen2.5vl:7b

# Single study (dev pipeline):
python pipeline_dev.py single pdfs/study59.pdf --study-id study59

# Full dev-set batch:
python pipeline_dev.py batch --pdf-dir pdfs/ \
    --articles-csv data/dev_articles.csv \
    --output-csv submission_dev.csv

# Single study (test pipeline, dynamic constructs):
python pipeline_test.py single pdfs/study1.pdf --study-id study1 \
    --articles-csv data/test_articles.csv \
    --construct-definitions-csv data/test_construct_definitions.csv

# Full test-set batch:
python pipeline_test.py batch --pdf-dir pdfs/ \
    --articles-csv data/test_articles.csv \
    --construct-definitions-csv data/test_construct_definitions.csv \
    --output-csv submission_test.csv
```

## Submission-time imputation

Grand-mean imputation is applied **after** the pipeline writes its submission CSV, not inside the pipeline. The pipeline writes blanks for studies it could not extract from; imputation fills the blanks. Sample PowerShell:

```powershell
$m = (Import-Csv submission_test.csv | Where-Object {$_.r -ne ''} |
      Measure-Object -Property r -Average).Average
(Import-Csv submission_test.csv) | ForEach-Object {
    if ($_.r -eq '') {$_.r = $m}; $_
} | Export-Csv final.csv -NoType
```

See [KNOWN_LANDMINES.md — Landmine 7](docs/KNOWN_LANDMINES.md#landmine-7--imputation-belongs-at-submission-not-in-the-pipeline).

**Follow-up:** Add `docs/one_hot_key_siop_2026.pdf` when publishing the deck; the link above 404s until then.

## Hardware

Built and benchmarked on a workstation named **Walt**:

- Intel Core Ultra 9 285HX (8P+16E cores, 24 threads)
- NVIDIA RTX PRO 3000 Blackwell Laptop GPU (12 GB GDDR7)
- 64 GB DDR5 RAM
- All inference runs locally; no API calls in the standard pipeline.

See [ARCHITECTURE.md section 7](docs/ARCHITECTURE.md) for library and VRAM rationale.

## Tests

```bash
SIOP_PIPELINE_V10_UNIT=1 python pipeline_dev.py
SIOP_PIPELINE_V12_DYN=1  python pipeline_test.py
```

Both should print an OK line and exit 0. There is no separate pytest harness.

## License & credits

- **Pipeline:** Matt Mongiello. License TBD before public release.
- **Models:** phi4 (Microsoft, MIT license), qwen2.5-VL:7B (Alibaba, Tongyi Qianwen license), served locally via Ollama.
- **Libraries:** Docling (IBM, MIT), pdfplumber (Jeremy Singer-Vine, MIT), PyMuPDF (Artifex Software, AGPL).
- **Agents:** Claude (Anthropic) and Cursor (Anysphere).

## Further reading

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — 4-tier cascade design rationale and library choices.
- [KNOWN_LANDMINES.md](docs/KNOWN_LANDMINES.md) — silent-failure traps to never re-introduce.
- [CHANGELOG.md](CHANGELOG.md) — version history v9 through v12.1.
- [META_ANALYSIS_MANUAL.md](docs/META_ANALYSIS_MANUAL.md) — adaptation guide for forking to a new research question.
