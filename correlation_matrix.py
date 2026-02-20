#!/usr/bin/env python3
"""
Correlation Matrix - KruegerAlgorithms
https://kruegeralgorithms.com

Calculates and visualizes correlation between multiple trading assets
using daily returns from Yahoo Finance.

Usage:
    python correlation_matrix.py
    python correlation_matrix.py --symbols ^GDAXI EURUSD=X GC=F ^IXIC
    python correlation_matrix.py --symbols ^GDAXI EURUSD=X --period 6mo --rolling 30
    python correlation_matrix.py --symbols ^GDAXI ^FTSE ^GSPC GC=F --output correlations.png
"""

import argparse
import sys
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance required. Install with: pip install yfinance")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: pandas and numpy required. Install with: pip install pandas numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visual output disabled.")
    print("Install with: pip install matplotlib")


# ─── DEFAULT SYMBOLS ───────────────────────────────────────────

DEFAULT_SYMBOLS = {
    '^GDAXI':   'DAX',
    '^FTSE':    'FTSE',
    '^GSPC':    'S&P 500',
    '^IXIC':    'NASDAQ',
    'EURUSD=X': 'EUR/USD',
    'GBPUSD=X': 'GBP/USD',
    'GC=F':     'Gold',
    'CL=F':     'Oil (WTI)',
}


# ─── DATA FETCHING ─────────────────────────────────────────────

def fetch_data(symbols: list, period: str = '1y') -> pd.DataFrame:
    """Fetch daily close prices for multiple symbols."""
    print(f"Fetching data for {len(symbols)} symbols (period: {period})...")

    data = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period=period)
            if len(hist) > 0:
                label = DEFAULT_SYMBOLS.get(sym, sym)
                data[label] = hist['Close']
                print(f"  {label} ({sym}): {len(hist)} days")
            else:
                print(f"  {sym}: No data found, skipping")
        except Exception as e:
            print(f"  {sym}: Error - {e}")

    if len(data) < 2:
        print("Error: Need at least 2 symbols with data")
        sys.exit(1)

    df = pd.DataFrame(data)
    df = df.dropna()  # Only keep dates where all symbols have data
    print(f"Aligned data: {len(df)} trading days")
    return df


# ─── CORRELATION ANALYSIS ─────────────────────────────────────

def calculate_correlations(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix from daily returns."""
    returns = prices.pct_change().dropna()
    return returns.corr()


def calculate_rolling_correlation(prices: pd.DataFrame, sym1: str, sym2: str,
                                   window: int = 30) -> pd.Series:
    """Calculate rolling correlation between two symbols."""
    returns = prices.pct_change().dropna()
    return returns[sym1].rolling(window).corr(returns[sym2])


# ─── VISUALIZATION ─────────────────────────────────────────────

def plot_heatmap(corr_matrix: pd.DataFrame, output_path: str):
    """Plot correlation heatmap."""
    if not HAS_MATPLOTLIB:
        return

    n = len(corr_matrix)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), max(6, n * 1.0)))

    # Custom colormap: red (negative) -> white (neutral) -> green (positive)
    colors = ['#d32f2f', '#ef5350', '#ffffff', '#66bb6a', '#2e7d32']
    cmap = LinearSegmentedColormap.from_list('rg', colors, N=256)

    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(corr_matrix.index, fontsize=10)

    # Add correlation values as text
    for i in range(n):
        for j in range(n):
            val = corr_matrix.iloc[i, j]
            text_color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=text_color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')
    ax.set_title('Asset Correlation Matrix (Daily Returns)', fontsize=14, fontweight='light', pad=15)

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"Heatmap saved: {output_path}")


def plot_rolling(prices: pd.DataFrame, pairs: list, window: int, output_path: str):
    """Plot rolling correlations for specified pairs."""
    if not HAS_MATPLOTLIB or not pairs:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    symbols = list(prices.columns)
    plotted = 0

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            rolling = calculate_rolling_correlation(prices, symbols[i], symbols[j], window)
            label = f'{symbols[i]} vs {symbols[j]}'
            ax.plot(rolling.index, rolling.values, linewidth=1.2, label=label, alpha=0.8)
            plotted += 1
            if plotted >= 6:  # Max 6 pairs to keep it readable
                break
        if plotted >= 6:
            break

    ax.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
    ax.axhline(y=0.7, color='green', linewidth=0.5, alpha=0.3, linestyle='--')
    ax.axhline(y=-0.7, color='red', linewidth=0.5, alpha=0.3, linestyle='--')

    ax.set_title(f'Rolling Correlation ({window}-day window)', fontsize=14, fontweight='light')
    ax.set_ylabel('Correlation')
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.2)

    rolling_path = output_path.replace('.png', '_rolling.png')
    fig.savefig(rolling_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"Rolling chart saved: {rolling_path}")


# ─── TEXT OUTPUT ───────────────────────────────────────────────

def print_matrix(corr_matrix: pd.DataFrame):
    """Print correlation matrix as formatted text."""
    symbols = list(corr_matrix.columns)
    n = len(symbols)

    # Header
    max_len = max(len(s) for s in symbols)
    header = " " * (max_len + 2)
    for s in symbols:
        header += f"{s:>10}"
    print("\n" + header)
    print("-" * len(header))

    # Rows
    for i, sym in enumerate(symbols):
        row = f"{sym:<{max_len + 2}}"
        for j in range(n):
            val = corr_matrix.iloc[i, j]
            row += f"{val:>10.3f}"
        print(row)

    # Highlight strong correlations
    print("\nStrong correlations (|r| > 0.7):")
    found = False
    for i in range(n):
        for j in range(i + 1, n):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.7:
                direction = "positive" if val > 0 else "negative"
                print(f"  {symbols[i]} <-> {symbols[j]}: {val:.3f} ({direction})")
                found = True
    if not found:
        print("  None found")


# ─── MAIN ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Asset Correlation Matrix')
    parser.add_argument('--symbols', '-s', nargs='+',
                        default=list(DEFAULT_SYMBOLS.keys()),
                        help='Yahoo Finance symbols (e.g., ^GDAXI EURUSD=X GC=F)')
    parser.add_argument('--period', '-p', default='1y',
                        help='Data period (1mo, 3mo, 6mo, 1y, 2y, 5y)')
    parser.add_argument('--rolling', '-r', type=int, default=30,
                        help='Rolling correlation window (days)')
    parser.add_argument('--output', '-o', default='correlation_matrix.png',
                        help='Output image path')
    parser.add_argument('--no-chart', action='store_true',
                        help='Skip chart generation')
    args = parser.parse_args()

    prices = fetch_data(args.symbols, args.period)
    corr_matrix = calculate_correlations(prices)

    print_matrix(corr_matrix)

    if not args.no_chart and HAS_MATPLOTLIB:
        plt.style.use('dark_background')
        plot_heatmap(corr_matrix, args.output)
        plot_rolling(prices, args.symbols, args.rolling, args.output)

    print("\nDone!")


if __name__ == '__main__':
    main()
