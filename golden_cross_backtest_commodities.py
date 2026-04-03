import yfinance as yf
import pandas as pd
import numpy as np

# --- Configuration ---
TICKERS = {
    "CRUD": "CRUD.L",   # WisdomTree WTI Crude Oil (LSE)
    "CARB": "CARB.L",   # WisdomTree Carbon
    "NGAS": "NGAS.L"    # WisdomTree Natural Gas
}

TOTAL_CAPITAL = 1000
CAPITAL_PER_TICKER = TOTAL_CAPITAL / len(TICKERS)

RISK_PER_TRADE = 0.02   # 2%
STOP_LOSS_PCT = 0.08    # 8%

SMA_SHORT = 50
SMA_LONG = 200

START_DATE = "2018-01-01"


# --- Data Download ---
def download_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START_DATE, auto_adjust=True)

    # Flatten multi-index columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].dropna()
    df["Close"] = df["Close"].astype(float)
    df["Returns"] = df["Close"].pct_change()
    df["Excess Returns"] = df["Returns"] - 0.04/252 # Daily return minus risk-free rate annualized
    valid_returns = df["Excess Returns"].dropna()
    sharpe = np.sqrt(252) * np.average(valid_returns) / np.std(valid_returns)
    print(f"The Sharpe ratio for a simple long-only strategy is equal to {sharpe:.2f}")
    return df


# --- Signal Calculation ---
def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sma_short"] = df["Close"].rolling(SMA_SHORT).mean()
    df["sma_long"] = df["Close"].rolling(SMA_LONG).mean()

    # Golden Cross
    df["golden_cross"] = (
        (df["sma_short"] > df["sma_long"]) &
        (df["sma_short"].shift(1) <= df["sma_long"].shift(1))
    )

    # Death Cross
    df["death_cross"] = (
        (df["sma_short"] < df["sma_long"]) &
        (df["sma_short"].shift(1) >= df["sma_long"].shift(1))
    )

    # Trend filter
    df["trend_filter"] = df["Close"] > df["sma_long"]

    return df


# --- Position Sizing ---
def compute_position_size(capital: float, price: float) -> int:
    risk_amount = capital * RISK_PER_TRADE
    risk_per_unit = price * STOP_LOSS_PCT

    units = risk_amount / risk_per_unit
    return max(1, int(units))


# --- Backtest ---
def backtest_strategy(name: str, df: pd.DataFrame) -> dict:
    capital = CAPITAL_PER_TICKER
    position = 0
    entry_price = 0
    stop_loss = 0

    trades = []

    for date, row in df.iterrows():
        price = float(row["Close"])

        # --- Stop loss ---
        if position > 0 and price <= stop_loss:
            pnl = (price - entry_price) * position
            capital += position * price

            trades.append({
                "exit_date": date,
                "type": "STOP",
                "exit_price": price,
                "pnl": round(pnl, 2)
            })

            position = 0

        # --- Entry ---
        if position == 0 and row["golden_cross"] and row["trend_filter"]:
            units = compute_position_size(capital, price)
            cost = units * price

            if cost <= capital:
                position = units
                entry_price = price
                stop_loss = price * (1 - STOP_LOSS_PCT)

                capital -= cost

                trades.append({
                    "entry_date": date,
                    "type": "BUY",
                    "entry_price": round(price, 2),
                    "units": units,
                    "stop_loss": round(stop_loss, 2)
                })

        # --- Exit ---
        elif position > 0 and row["death_cross"]:
            pnl = (price - entry_price) * position
            capital += position * price

            trades.append({
                "exit_date": date,
                "type": "SELL",
                "exit_price": price,
                "pnl": round(pnl, 2)
            })

            position = 0

    # --- Final portfolio value ---
    final_value = capital
    if position > 0:
        final_value += position * float(df["Close"].iloc[-1])

    # --- Metrics ---
    total_return = (final_value - CAPITAL_PER_TICKER) / CAPITAL_PER_TICKER * 100

    closed_trades = [t for t in trades if "pnl" in t]
    winning_trades = [t for t in closed_trades if t["pnl"] > 0]

    win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0

    return {
        "instrument": name,
        "final_capital": round(final_value, 2),
        "return_pct": round(total_return, 2),
        "num_trades": len(closed_trades),
        "win_rate": round(win_rate, 1),
        "trades": trades
    }


# --- Results Display ---
def display_results(results: list):
    print("\n" + "=" * 55)
    print("BACKTEST RESULTS — Golden Cross Strategy")
    print("=" * 55)

    for r in results:
        print(f"\n{r['instrument']}")
        print(f"  Initial capital : {CAPITAL_PER_TICKER:.0f}€")
        print(f"  Final capital   : {r['final_capital']}€")
        print(f"  Return          : {r['return_pct']:+.1f}%")
        print(f"  Number of trades: {r['num_trades']}")
        print(f"  Win rate        : {r['win_rate']}%")

        closed_trades = [t for t in r["trades"] if "pnl" in t]

        if closed_trades:
            pnls = [t["pnl"] for t in closed_trades]
            gains = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            rr_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else 0

            print(f"  Avg gain        : +{avg_gain:.1f}€")
            print(f"  Avg loss        : {avg_loss:.1f}€")
            print(f"  Risk/Reward     : {rr_ratio:.1f}x")

    print("\n" + "=" * 55)


# --- Main ---
if __name__ == "__main__":
    results = []

    for name, ticker in TICKERS.items():
        print(f"\nDownloading {name} ({ticker})...")

        df = download_data(ticker)

        # Debug (optional)
        print(df.tail())

        df = compute_signals(df)

        result = backtest_strategy(name, df)
        results.append(result)

    display_results(results)