# ML Insight Lab — Interactive UI Update

This is a drop-in UI update for an existing ML capstone project. It keeps the integrated Overview, Classification, Regression, Clustering, Dataset Explorer, and Final Decision pages.

## Install into your existing project

1. Back up your current `app/app.py`.
2. Copy this package's `app/app.py` over your existing project's `app/app.py`.
3. Keep your existing `data/`, `notebooks/`, and `models/` folders. Do not delete or replace them.
4. From the project root, run:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Dataset files expected

- `data/MarketArrivals.csv` for the clustering explorer.
- `data/Bengaluru_House_Data.csv` (or a filename containing `Bengaluru` and `House`) for regression prediction diagnostics.
- Other CSV files in `data/` are listed in Dataset Explorer.

Classification comparison values and regression comparison/cross-validation values are taken from the current notebooks. Regression diagnostics are recalculated from the housing CSV when it is available. The dashboard does not need to replace your notebooks.

## Design

The UI uses a restrained warm-neutral background, deep slate headings, muted sage accents, spacious cards, clear metric labels, interactive metric selectors, charts, tabs, filters, and expanders.
