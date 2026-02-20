# Correlation Matrix (Python)

Python CLI tool that calculates and visualizes the correlation between multiple trading assets using daily returns from Yahoo Finance, with heatmap and rolling correlation charts.

## Features

- **Yahoo Finance Data** - Fetches daily close prices for any Yahoo Finance symbol
- **Correlation Heatmap** - Color-coded matrix (red = negative, green = positive) with values
- **Rolling Correlation** - Time-series chart of rolling correlation between pairs
- **Default Symbols** - Pre-configured with DAX, FTSE, S&P 500, NASDAQ, EUR/USD, GBP/USD, Gold, Oil
- **Text Output** - Prints matrix to console with strong correlation highlighting
- **Configurable Period** - Data period from 1 month to 5 years

## Requirements

```
pip install yfinance pandas numpy matplotlib
```

## Usage

```bash
python correlation_matrix.py
python correlation_matrix.py --symbols ^GDAXI EURUSD=X GC=F ^IXIC
python correlation_matrix.py --symbols ^GDAXI ^FTSE --period 6mo --rolling 30
python correlation_matrix.py --output correlations.png --no-chart
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| --symbols, -s | DAX, FTSE, S&P, NASDAQ, EUR/USD, GBP/USD, Gold, Oil | Yahoo Finance symbols |
| --period, -p | 1y | Data period (1mo, 3mo, 6mo, 1y, 2y, 5y) |
| --rolling, -r | 30 | Rolling correlation window in days |
| --output, -o | correlation_matrix.png | Output image path |
| --no-chart | false | Skip chart generation (text only) |

## License

MIT License - Free to use, modify and distribute.

## Author

[KruegerAlgorithms](https://kruegeralgorithms.com)
