# ETH Drawdown Prediction with LSTM

This university machine learning project predicts short-term Ethereum drawdowns.

## Project Goal

We build a binary classifier that predicts whether ETH will experience a drawdown in the next 6 hours.

- Target label at hour t:
	- label = 1 if ETH drops by 3% or more in the next 6-hour interval
	- label = 0 otherwise

We use an LSTM model over rolling 24-hour sequences of hourly features.

## Data Sources

- Kaggle CSV: hourly ETH market data (OHLCV)
- Google BigQuery Ethereum public datasets: whale behavior features (large transfers, exchange inflows, gas usage, stablecoin movement)

Final modeling table is one row per hour containing:

- timestamp
- price features
- whale behavior features
- binary drawdown label

## Repository Layout

```
ml-crypto-project/
├─ data/
│  ├─ raw/          # Kaggle CSV, BigQuery exports (large; usually not committed)
│  └─ processed/    # merged + labeled data
├─ notebooks/
│  ├─ 01_exploration.ipynb
│  ├─ 02_merge_and_label.ipynb
│  └─ 03_plots.ipynb
├─ src/
│  ├─ data_prep.py
│  └─ models.py
├─ reports/
│  └─ figures/
├─ requirements.txt
└─ README.md
```

## Environment Setup (macOS + VS Code)

1. Clone and enter the repository
2. Create and activate a local virtual environment
3. Install dependencies

Commands:

```bash
git clone <your-repo-url>
cd ML-Project-2026

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## VS Code Notebook Kernel

In VS Code:

1. Open `notebooks/01_exploration.ipynb`
2. Click `Select Kernel`
3. Choose the interpreter from `.venv`

## How to Run the First Exploration Notebook

1. Put your ETH CSV in `data/raw/` (for example `data/raw/eth_hourly.csv`)
2. Open `notebooks/01_exploration.ipynb`
3. Run all cells to:
	 - load data
	 - parse timestamp
	 - filter ETH if needed
	 - resample hourly if needed
	 - plot ETH close price over time

## Notes

- Keep large raw files out of Git when possible.
- Store intermediate processed files in `data/processed/`.
- Put report plots in `reports/figures/`.
