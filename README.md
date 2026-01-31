# Transaction Classification & Spending Analysis

This repository contains two simple Python scripts to classify bank transactions with an LLM and analyze spending:

- `transaction-classification.py` — classifies rows in `mastersheet.csv` into categories using the OpenAI API and caches results in `classification_cache.json`.
- `spending-analysis.py` — generates summary stats and charts from the classified output `mastersheet_classified.csv`.

Setup
 - Create a `.env` file (or set environment variables) with your API key. See `.env.example`.
 - Optionally install `python-dotenv` to load `.env` automatically:
 - `mastersheet.csv` need 'Description' and 'Amount' Column
 - Assume amount is positive number

```
pip install python-dotenv
```"C:\Users\thoma\Projects"
```
Usage
- Classify transactions:

```
python transaction-classification.py
```

This reads `mastersheet.csv`, writes `mastersheet_classified.csv`, and caches results to `classification_cache.json`.

- Analyze spending (after classification):

```
python spending-analysis.py
```

Outputs
- `mastersheet_classified.csv` — classified transactions
- `classification_cache.json` — cache of previous classifications
- `chart_spend_by_category.png`, `chart_monthly_spend.png`, `chart_category_share.png` — analysis charts
