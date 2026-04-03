import pandas as pd
import numpy as np
import yfinance as yf

# --- Configuration ---
TICKERS = {
    "Apple": "AAPL",   # Apple
    "S&P500 ETF": "SPY",   # S&P 500 ETF
}

START_DATE = "2018-01-01"

# --- Data Download ---
def download_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START_DATE, auto_adjust=True)

    # Flatten multi-index columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    return df

# --- Sharpe ratio computation ---
def compute_sharpe_ratio(df: pd.DataFrame, rf_annual=0.04) -> float:
    rf_daily = rf_annual / 252
    df = df.copy()
    df["Returns_AAPL"] = df["AAPL"].pct_change()
    df["Returns_SPY"] = df["SPY"].pct_change()
    df["Strategy_Returns"] = (df["Returns_AAPL"] - df["Returns_SPY"])/2
    df = df.dropna()

    excess_returns = df["Strategy_Returns"] # No risk-free return since long-short is almost market-neutral

    if excess_returns.std() == 0:
        return 0

    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    return sharpe

# --- Max drawdown and Max drawdown duration computation ---
def compute_drawdown(df: pd.DataFrame) -> float:
    df = df.copy()
    df["Returns_AAPL"] = df["AAPL"].pct_change()
    df["Returns_SPY"] = df["SPY"].pct_change()
    df["Strategy_Returns"] = (df["Returns_AAPL"] - df["Returns_SPY"])/2
    df["Cumulative_Returns"] = (1+df["Strategy_Returns"]).cumprod()-1
    df["Equity_Curve"] = 1 + df["Cumulative_Returns"]
    df["High_Watermark"] = df["Equity_Curve"].cummax()
    df["Drawdown"] = df["Equity_Curve"]/df["High_Watermark"]-1
    df["Drawdown_Duration"] = (df["Drawdown"] < 0).astype(int)
    df["Drawdown_Duration"] = df["Drawdown_Duration"] * (df["Drawdown_Duration"].groupby((df["Drawdown_Duration"] == 0).cumsum()).cumcount())

    df = df.dropna()

    max_drawdown = df["Drawdown"].min()
    max_drawdown_duration = df["Drawdown_Duration"].max()

    return max_drawdown, max_drawdown_duration


df_aapl = download_data("AAPL")
df_spy = download_data ("SPY")

df_merge = pd.concat([df_aapl["Close"], df_spy["Close"]], axis = 1)
df_merge = df_merge.dropna()

df_merge.columns = ["AAPL", "SPY"]
print(df_merge.head())

sharpe_longshort = compute_sharpe_ratio(df_merge)
[maximum_drawdown, maximum_drawdown_duration] = compute_drawdown(df_merge)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SHARPE RATIO — Long AAPL / Short SPY strategy")
    print("="*60)
    print(f"→ Sharpe: {sharpe_longshort:.2f}")
    
    print("\n" + "="*60)
    print(" Maximum drawdown & Maximum drawdown duration — Long AAPL / Short SPY strategy")
    print("="*60)
    print(f"→ Maximum drawdown: {maximum_drawdown:.2f}")
    print(f"→ Maximum drawdown duration: {maximum_drawdown_duration}")
