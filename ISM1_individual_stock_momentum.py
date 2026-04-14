"""
ISM1 — Individual Stock Momentum
===================================
Universe     : ~100 large-cap US stocks across all sectors
Signal       : 12-1 month total return (skip last month to avoid short-term reversal)
Portfolio    : Equal-weight top 20 stocks by momentum score
Filter       : Absolute — only hold stocks with positive 12-1m momentum
Rebalance    : Monthly (last trading day of each month)
Commission   : 5 bps per leg on stocks entering / exiting the portfolio

Academic basis: Jegadeesh & Titman (1993) — stocks that performed best over
the past 12 months tend to continue outperforming over the next 1-12 months.

WARNING: This backtest uses a fixed universe of *current* large-cap stocks.
Survivorship bias is present — delisted or bankrupt stocks are excluded.
Expect real live performance to be ~1-3% lower than backtested results.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Universe ───────────────────────────────────────────────────────────────────
UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CSCO",
    "INTC", "TXN", "QCOM", "IBM", "AMD", "AMAT", "ADI", "MU", "NOW",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "V", "MA",
    "USB", "PNC", "SCHW", "COF", "MET", "TRV", "CB",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABT", "MRK", "LLY", "BMY", "AMGN", "MDT",
    "CVS", "CI", "HUM", "ISRG", "SYK", "BSX", "ZTS", "VRTX", "REGN",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "CL", "MDLZ",
    # Consumer Discretionary
    "AMZN", "TSLA", "MCD", "SBUX", "NKE", "HD", "LOW", "TGT", "TJX",
    "BKNG", "MAR", "F", "GM",
    # Industrials
    "HON", "CAT", "GE", "UPS", "FDX", "BA", "LMT", "RTX", "NOC",
    "DE", "MMM", "EMR", "ETN", "ITW",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "MPC", "OXY",
    # Materials
    "LIN", "APD", "ECL", "NEM", "FCX", "DD",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP",
    # Communication
    "T", "VZ", "CMCSA", "DIS", "NFLX",
    # Real Estate
    "AMT", "PLD", "CCI", "SPG", "EQIX",
]

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "universe"    : UNIVERSE,
    "start"       : "2005-01-01",
    "end"         : "2024-12-31",
    "lookback"    : 12,          # months for momentum signal
    "n_positions" : 20,          # number of stocks held at once
    "commission"  : 0.0005,      # 5 bps per leg
}

# OOS parameters identical — no re-fitting.
# Start early enough to cover the 13-month warmup before 2025-01-01.
OOS_CONFIG = {
    **CONFIG,
    "start"     : "2023-01-01",
    "end"       : "2026-04-12",
    "oos_start" : "2025-01-01",
}


# ── Data ──────────────────────────────────────────────────────────────────────
def download_data(cfg):
    """Download adjusted closes for universe + SPY + AGG, resample to month-end."""
    all_tickers = cfg["universe"] + ["SPY", "AGG"]
    print(f"  Downloading {len(all_tickers)} tickers from {cfg['start']} to {cfg['end']} ...")
    prices = yf.download(
        all_tickers, start=cfg["start"], end=cfg["end"],
        auto_adjust=True, progress=False
    )["Close"]
    prices = prices.ffill()
    monthly = prices.resample("ME").last()
    return monthly


# ── Signal ────────────────────────────────────────────────────────────────────
def compute_signals(monthly, cfg):
    """
    At each month-end t:
      - 12-1 momentum = price[t-1] / price[t-13] - 1  (skip last month)
      - Rank all universe stocks by this score
      - Select top N with positive score
    Returns dict { signal_date : [list of tickers to hold next month] }
    """
    lb   = cfg["lookback"]       # 12
    n    = cfg["n_positions"]    # 20
    uni  = [t for t in cfg["universe"] if t in monthly.columns]

    # 12-1 month momentum: shift(1) skips last month, pct_change(lb) looks back lb months
    mom = monthly[uni].shift(1).pct_change(lb)

    signals = {}
    for i in range(lb + 1, len(monthly)):
        date = monthly.index[i]
        row  = mom.iloc[i].dropna()
        row  = row[row > 0]                             # absolute momentum filter
        top  = row.nlargest(min(n, len(row))).index.tolist()
        signals[date] = top

    return signals


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(cfg):
    """
    Monthly backtest: signal at month-end M → equal-weight portfolio held in M+1.
    Commission charged proportional to turnover (stocks entering / exiting).
    Returns (equity, bench_spy, bench_6040, port_ret, signals, n_held_series).
    """
    monthly  = download_data(cfg)
    signals  = compute_signals(monthly, cfg)
    mret     = monthly.pct_change()

    port_rets, port_dates, n_held = [], [], []
    prev_holdings = set()

    for sig_date, holdings in signals.items():
        future = mret.index[mret.index > sig_date]
        if len(future) == 0:
            break
        next_date = future[0]

        if len(holdings) == 0:
            port_rets.append(0.0)
            port_dates.append(next_date)
            n_held.append(0)
            prev_holdings = set()
            continue

        # Equal-weight return
        stock_rets = mret.loc[next_date, holdings].dropna()
        ret = float(stock_rets.mean()) if len(stock_rets) > 0 else 0.0

        # Commission on turnover
        curr = set(holdings)
        changes = len(curr - prev_holdings) + len(prev_holdings - curr)
        n_pos   = max(len(curr), 1)
        turnover = changes / (2 * n_pos)
        ret -= turnover * cfg["commission"] * 2

        port_rets.append(ret)
        port_dates.append(next_date)
        n_held.append(len(curr))
        prev_holdings = curr

    idx      = pd.DatetimeIndex(port_dates)
    port_ret = pd.Series(port_rets, index=idx, name="strategy")
    equity   = (1 + port_ret).cumprod()

    spy_ret    = mret["SPY"].reindex(idx)
    bench_spy  = (1 + spy_ret).cumprod()
    ret_6040   = 0.6 * mret["SPY"] + 0.4 * mret["AGG"]
    bench_6040 = (1 + ret_6040.reindex(idx)).cumprod()
    n_held_s   = pd.Series(n_held, index=idx)

    return equity, bench_spy, bench_6040, port_ret, signals, n_held_s


# ── OOS test ──────────────────────────────────────────────────────────────────
def run_oos_test(cfg):
    """Identical signal logic applied to 2025-2026 data — zero re-fitting."""
    monthly  = download_data(cfg)
    signals_all = compute_signals(monthly, cfg)
    mret     = monthly.pct_change()

    port_rets, port_dates, n_held = [], [], []
    prev_holdings = set()

    for sig_date, holdings in signals_all.items():
        future = mret.index[mret.index > sig_date]
        if len(future) == 0:
            break
        next_date = future[0]

        if len(holdings) == 0:
            port_rets.append(0.0)
            port_dates.append(next_date)
            n_held.append(0)
            prev_holdings = set()
            continue

        stock_rets = mret.loc[next_date, holdings].dropna()
        ret = float(stock_rets.mean()) if len(stock_rets) > 0 else 0.0

        curr = set(holdings)
        changes = len(curr - prev_holdings) + len(prev_holdings - curr)
        n_pos   = max(len(curr), 1)
        ret -= (changes / (2 * n_pos)) * cfg["commission"] * 2

        port_rets.append(ret)
        port_dates.append(next_date)
        n_held.append(len(curr))
        prev_holdings = curr

    idx      = pd.DatetimeIndex(port_dates)
    port_all = pd.Series(port_rets, index=idx, name="strategy")
    n_all    = pd.Series(n_held, index=idx)

    oos_start  = pd.Timestamp(cfg["oos_start"])
    port_ret   = port_all[port_all.index >= oos_start]
    n_held_s   = n_all[n_all.index >= oos_start]

    # Signals trimmed to OOS window (for display)
    sig_series = pd.Series({d: v for d, v in signals_all.items()
                            if d >= oos_start})

    equity     = (1 + port_ret).cumprod()
    spy_ret    = mret["SPY"].reindex(port_ret.index)
    bench_spy  = (1 + spy_ret).cumprod()
    ret_6040   = 0.6 * mret["SPY"] + 0.4 * mret["AGG"]
    bench_6040 = (1 + ret_6040.reindex(port_ret.index)).cumprod()

    return equity, bench_spy, bench_6040, port_ret, sig_series, n_held_s, cfg


# ── Statistics ────────────────────────────────────────────────────────────────
def _metrics(monthly_rets, label=""):
    eq   = (1 + monthly_rets).cumprod()
    n    = len(monthly_rets)
    cagr = eq.iloc[-1] ** (12 / n) - 1
    vol  = monthly_rets.std() * np.sqrt(12)
    sr   = cagr / vol if vol > 0 else 0.0
    dd   = (eq - eq.cummax()) / eq.cummax()
    mdd  = dd.min()
    cal  = cagr / abs(mdd) if mdd < 0 else float("nan")
    return dict(label=label, cagr=cagr, vol=vol, sharpe=sr, mdd=mdd, calmar=cal)


def _print_stats(equity, bench_spy, bench_6040, port_ret, cfg):
    spy_ret  = bench_spy.pct_change().dropna()
    ret_6040 = bench_6040.pct_change().dropna()

    rows = [
        _metrics(port_ret, "ISM Strategy"),
        _metrics(spy_ret,  "SPY B&H"),
        _metrics(ret_6040, "60/40"),
    ]

    print("\n" + "═" * 72)
    print(f"  INDIVIDUAL STOCK MOMENTUM — {cfg['start']} → {cfg['end']}")
    print("═" * 72)
    print(f"  {'':20s} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8}")
    print("  " + "─" * 60)
    for r in rows:
        print(f"  {r['label']:20s} {r['cagr']:>8.1%} {r['vol']:>8.1%} "
              f"{r['sharpe']:>8.2f} {r['mdd']:>8.1%} {r['calmar']:>8.2f}")
    print("  " + "─" * 60)

    # Per-year table
    print(f"\n  {'Year':>6}  {'ISM':>8}  {'SPY':>8}  {'60/40':>8}")
    print("  " + "─" * 36)
    for yr in sorted(set(port_ret.index.year)):
        gm_r = (1 + port_ret[port_ret.index.year == yr]).prod() - 1
        sy_r = (1 + spy_ret[spy_ret.index.year == yr]).prod() - 1
        s6_r = (1 + ret_6040[ret_6040.index.year == yr]).prod() - 1
        beat = " ✓" if gm_r > sy_r else ""
        print(f"  {yr:>6}  {gm_r:>8.1%}  {sy_r:>8.1%}  {s6_r:>8.1%}{beat}")
    print("  " + "─" * 36)
    print("═" * 72)


def _print_oos_stats(bench_spy, bench_6040, port_ret):
    spy_ret  = bench_spy.pct_change().dropna()
    ret_6040 = bench_6040.pct_change().dropna()

    for yr in sorted(set(port_ret.index.year)):
        gm_yr = port_ret[port_ret.index.year == yr]
        sy_yr = spy_ret[spy_ret.index.year == yr]
        s6_yr = ret_6040[ret_6040.index.year == yr]

        is_partial = (yr == port_ret.index[-1].year and
                      port_ret.index[-1].month < 12)
        label = f"{yr} YTD" if is_partial else str(yr)

        rows = [
            _metrics(gm_yr, "ISM Strategy"),
            _metrics(sy_yr, "SPY B&H"),
            _metrics(s6_yr, "60/40"),
        ]

        print("\n\n")
        print("█" * 72)
        print(f"  TEST SET — {label}  (OUT-OF-SAMPLE — zero re-fitting)")
        print("█" * 72)
        print(f"  {'':20s} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8}")
        print("  " + "─" * 60)
        for r in rows:
            print(f"  {r['label']:20s} {r['cagr']:>8.1%} {r['vol']:>8.1%} "
                  f"{r['sharpe']:>8.2f} {r['mdd']:>8.1%} {r['calmar']:>8.2f}")
        print("  " + "─" * 60)

        # Monthly detail
        print(f"\n  {'Month':>10}  {'ISM':>8}  {'SPY':>8}")
        print("  " + "─" * 30)
        for mo_date in gm_yr.index:
            gm_mo = gm_yr.loc[mo_date]
            sy_mo = sy_yr.loc[mo_date] if mo_date in sy_yr.index else float("nan")
            beat  = " ✓" if gm_mo > sy_mo else ""
            print(f"  {mo_date.strftime('%Y-%m'):>10}  {gm_mo:>8.1%}  {sy_mo:>8.1%}{beat}")
        print("  " + "─" * 30)
        print("█" * 72)


# ── Plot — training ───────────────────────────────────────────────────────────
def _plot(equity, bench_spy, bench_6040, port_ret, n_held_s, cfg):
    c_bg  = "#0d1117"; c_gr = "#21262d"
    c_s   = "#58a6ff"; c_b  = "#f78166"; c_640 = "#3fb950"
    c_pos = "#3fb950"; c_neg = "#f85149"

    plt.rcParams.update({
        "figure.facecolor": c_bg, "axes.facecolor": c_bg,
        "axes.labelcolor": "white", "xtick.color": "white",
        "ytick.color": "white", "text.color": "white",
        "axes.titlecolor": "white",
    })

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    # Cumulative returns
    ax1.plot(equity.index,     equity,     color=c_s,   lw=2.2, label="ISM Strategy")
    ax1.plot(bench_spy.index,  bench_spy,  color=c_b,   lw=1.5, label="SPY B&H",  alpha=0.85)
    ax1.plot(bench_6040.index, bench_6040, color=c_640, lw=1.5, label="60/40",    alpha=0.75, ls="--")
    ax1.axhline(1, color=c_gr, ls="--", lw=0.8)
    ax1.set_title("Cumulative Returns", fontsize=11, weight="bold")
    ax1.set_ylabel("Portfolio Value ($1 invested)", color="white")
    ax1.legend(facecolor=c_bg, edgecolor=c_gr, labelcolor="white")
    ax1.grid(True, color=c_gr, lw=0.5)

    # Drawdown
    for eq, col, lbl in [(equity, c_s, "ISM"), (bench_spy, c_b, "SPY")]:
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax2.fill_between(dd.index, dd, 0, color=col, alpha=0.25)
        ax2.plot(dd.index, dd, color=col, lw=1.2, label=lbl)
    ax2.set_title("Drawdown (%)", fontsize=11, weight="bold")
    ax2.set_ylabel("%", color="white")
    ax2.legend(facecolor=c_bg, edgecolor=c_gr, labelcolor="white", fontsize=9)
    ax2.grid(True, color=c_gr, lw=0.5)

    # Annual returns
    spy_ret = bench_spy.pct_change().dropna()
    all_yrs = sorted(set(port_ret.index.year))
    yr_ism  = [(1 + port_ret[port_ret.index.year == yr]).prod() - 1 for yr in all_yrs]
    yr_spy  = [(1 + spy_ret[spy_ret.index.year == yr]).prod() - 1   for yr in all_yrs]

    x, wid = np.arange(len(all_yrs)), 0.38
    ax3.bar(x - wid/2, [r * 100 for r in yr_ism], wid,
            color=[c_pos if r >= 0 else c_neg for r in yr_ism], alpha=0.85, label="ISM")
    ax3.bar(x + wid/2, [r * 100 for r in yr_spy], wid,
            color=c_b, alpha=0.45, label="SPY")
    ax3.set_xticks(x)
    ax3.set_xticklabels(all_yrs, rotation=45, fontsize=7, color="white")
    ax3.axhline(0, color="white", lw=0.8, alpha=0.6)
    ax3.set_title("Annual Returns (%)", fontsize=11, weight="bold")
    ax3.set_ylabel("%", color="white")
    ax3.legend(facecolor=c_bg, edgecolor=c_gr, labelcolor="white", fontsize=8)
    ax3.grid(True, color=c_gr, lw=0.5, axis="y")

    # Monthly heatmap
    mo = (port_ret * 100).to_frame("r")
    mo["y"] = mo.index.year; mo["m"] = mo.index.month
    pv = mo.pivot(index="y", columns="m", values="r")
    pv.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"][:pv.shape[1]]
    sns.heatmap(pv, ax=ax4, cmap="RdYlGn", center=0, annot=True, fmt=".1f",
                linewidths=0.3, linecolor=c_bg, cbar=False,
                annot_kws={"size": 7, "color": "white"})
    ax4.set_title("Monthly Returns (%)", fontsize=11, weight="bold")
    ax4.tick_params(axis="x", rotation=0); ax4.set_xlabel("")

    # Rolling Sharpe (90-day using daily approximation from monthly)
    roll = port_ret.rolling(12).apply(
        lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0)
    ax5.plot(roll.index, roll, color=c_s, lw=1.2)
    ax5.axhline(0, color="white", ls="--", lw=0.8, alpha=0.5)
    ax5.axhline(1, color=c_pos,  ls="--", lw=0.8, alpha=0.7)
    ax5.fill_between(roll.index, roll, 0,
                     where=(roll >= 0), color=c_pos, alpha=0.15)
    ax5.fill_between(roll.index, roll, 0,
                     where=(roll < 0),  color=c_neg, alpha=0.15)
    ax5.set_title("Rolling 12-Month Sharpe", fontsize=11, weight="bold")
    ax5.grid(True, color=c_gr, lw=0.5)

    fig.suptitle(
        f"Individual Stock Momentum (ISM1)  |  {cfg['start']} – {cfg['end']}",
        fontsize=14, weight="bold", color="white", y=1.01)

    out = "/Users/user/Desktop/QuantitativeTrading/Quantitative_trading_exercises/ISM1_train.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=c_bg)
    print(f"Training chart saved → {out}")


# ── Plot — OOS ────────────────────────────────────────────────────────────────
def _plot_oos(equity, bench_spy, bench_6040, port_ret, n_held_s, cfg):
    c_bg  = "#0d1117"; c_gr = "#21262d"
    c_s   = "#58a6ff"; c_b  = "#f78166"; c_640 = "#3fb950"
    c_pos = "#3fb950"; c_neg = "#f85149"

    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Cumulative returns with monthly labels
    ax1.plot(equity.index,     equity,     color=c_s,   lw=2.2, label="ISM Strategy")
    ax1.plot(bench_spy.index,  bench_spy,  color=c_b,   lw=1.5, label="SPY B&H",  alpha=0.85)
    ax1.plot(bench_6040.index, bench_6040, color=c_640, lw=1.5, label="60/40",    alpha=0.75, ls="--")
    ax1.axhline(1, color=c_gr, ls="--", lw=0.8)
    for date, ret in port_ret.items():
        col = c_pos if ret >= 0 else c_neg
        ax1.annotate(f"{ret*100:.1f}%",
                     xy=(date, equity.loc[date]),
                     xytext=(0, 10), textcoords="offset points",
                     fontsize=6.5, color=col, ha="center")
    ax1.set_title("Cumulative Returns — Out-of-Sample", fontsize=11, weight="bold")
    ax1.set_ylabel("Portfolio Value ($1 invested)", color="white")
    ax1.legend(facecolor=c_bg, edgecolor=c_gr, labelcolor="white")
    ax1.grid(True, color=c_gr, lw=0.5)

    # Drawdown
    for eq, col, lbl in [(equity, c_s, "ISM"), (bench_spy, c_b, "SPY")]:
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax2.fill_between(dd.index, dd, 0, color=col, alpha=0.25)
        ax2.plot(dd.index, dd, color=col, lw=1.2, label=lbl)
    ax2.set_title("Drawdown (%)", fontsize=11, weight="bold")
    ax2.set_ylabel("%", color="white")
    ax2.legend(facecolor=c_bg, edgecolor=c_gr, labelcolor="white", fontsize=9)
    ax2.grid(True, color=c_gr, lw=0.5)

    # Number of stocks held each month
    ax3.bar(n_held_s.index, n_held_s.values, width=20,
            color=c_s, alpha=0.75)
    ax3.axhline(cfg["n_positions"], color=c_pos, ls="--", lw=1,
                label=f"Target ({cfg['n_positions']})")
    ax3.set_title("Stocks Held Per Month", fontsize=11, weight="bold")
    ax3.set_ylabel("Count", color="white")
    ax3.set_ylim(0, cfg["n_positions"] + 5)
    ax3.legend(facecolor=c_bg, edgecolor=c_gr, labelcolor="white", fontsize=8)
    ax3.grid(True, color=c_gr, lw=0.5, axis="y")

    fig.suptitle(
        f"Individual Stock Momentum (ISM1) — Out-of-Sample Test\n"
        f"Strategy designed on 2005–2024  |  Tested on {cfg['oos_start']} – {cfg['end']}",
        fontsize=13, weight="bold", color="white", y=1.02)

    out = "/Users/user/Desktop/QuantitativeTrading/Quantitative_trading_exercises/ISM1_oos.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=c_bg)
    print(f"OOS chart saved → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────
# Order: compute all → print all tables → save all plots → plt.show()
if __name__ == "__main__":
    print(__doc__)

    # ── 1. Compute ─────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  COMPUTING — training set (2005–2024) ...")
    print("=" * 72)
    tr_eq, tr_spy, tr_6040, tr_ret, tr_sig, tr_n = run_backtest(CONFIG)

    print("\n" + "=" * 72)
    print("  COMPUTING — out-of-sample test (2025–2026) ...")
    print("=" * 72)
    oos_eq, oos_spy, oos_6040, oos_ret, oos_sig, oos_n, oos_cfg = run_oos_test(OOS_CONFIG)

    # ── 2. Print all tables ────────────────────────────────────────────────────
    print("\n\n")
    print("█" * 72)
    print("  TABLE 1 — TRAINING SET RESULTS  (2005–2024, in-sample)")
    print("█" * 72)
    _print_stats(tr_eq, tr_spy, tr_6040, tr_ret, CONFIG)

    _print_oos_stats(oos_spy, oos_6040, oos_ret)

    # ── 3. Save charts ─────────────────────────────────────────────────────────
    print("\n\nSaving charts ...")
    _plot(tr_eq, tr_spy, tr_6040, tr_ret, tr_n, CONFIG)
    _plot_oos(oos_eq, oos_spy, oos_6040, oos_ret, oos_n, oos_cfg)

    plt.show()
