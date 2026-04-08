# CRIS

CRIS is a Python-first implementation scaffold for the NCRB retrieval, extraction, and crime-rate prediction pipeline described in the project notes already present in this repository.

## Stack

- Python 3.13
- `pdfplumber` for PDF and table extraction
- `chromadb` for local retrieval indexes
- `sentence-transformers` for embeddings
- `pandas` and `scikit-learn` for feature engineering and evaluation
- `xgboost` and `shap` for prediction and explanation
- `fastapi` and `typer` for API and CLI surfaces

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
python -m spacy download en_core_web_sm
cris --help
```

## Layout

- `src/cris/` application package
- `configs/` pipeline configuration
- `data/raw/` source NCRB PDFs
- `artifacts/` generated models, indexes, and reports
- `tests/` automated tests

## Current Status

This commit sets up the Python project structure and local environment conventions. The end-to-end NCRB implementation still needs to be built into the package modules.
