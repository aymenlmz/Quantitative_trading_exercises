"""
ISM1 — Interactive Brokers Paper Trading Deployment
=====================================================
Runs the Individual Stock Momentum strategy live against an IB paper
trading account.  Execute this script once on the last trading day of
each month (or automate it via cron / Task Scheduler).

Requirements
------------
    pip install ib_insync yfinance pandas numpy

IB setup
--------
    1. Open TWS or IB Gateway and log in to your **paper** trading account.
    2. In TWS:  File → Global Configuration → API → Settings
       • Enable "ActiveX and Socket Clients"
       • Port: 7497  (TWS paper) or 4002  (Gateway paper)
       • Uncheck "Read-Only API"
    3. Run this script from a machine on the same network as TWS/Gateway.

Safety controls
---------------
    DRY_RUN = True   → prints the intended trades but places NO orders.
    DRY_RUN = False  → places live orders on your paper account.

Always review the printed trade plan before flipping DRY_RUN to False.
"""

import time
import logging
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf
from ib_insync import IB, Stock, MarketOrder, util

# ── User configuration ────────────────────────────────────────────────────────
IB_HOST    = "127.0.0.1"
IB_PORT    = 7497          # 7497 = TWS paper | 4002 = Gateway paper
IB_CLIENT  = 1             # any integer; must be unique per simultaneous connection
DRY_RUN    = False         # KEEP True until you have reviewed a full trade plan

# Strategy parameters — must match the backtest exactly
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

LOOKBACK     = 12   # months for momentum signal
N_POSITIONS  = 20   # max stocks held at once

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ISM1")


# ── Signal computation ────────────────────────────────────────────────────────
def compute_target_portfolio() -> list[str]:
    """
    Download ~14 months of monthly price data and return the list of tickers
    the strategy wants to hold this month.

    Logic mirrors the backtest exactly:
      momentum[t] = price[t-1] / price[t-13] - 1   (skip last month)
      Select top N_POSITIONS with positive momentum.
    """
    # Need 14 months of data to compute the 12-1 signal at the latest month-end
    start = pd.Timestamp.today() - pd.DateOffset(months=15)
    log.info("Downloading price data for %d tickers (from %s) ...",
             len(UNIVERSE), start.strftime("%Y-%m-%d"))

    daily = yf.download(
        UNIVERSE, start=start.strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False
    )["Close"]
    daily = daily.ffill()
    monthly = daily.resample("ME").last()

    if len(monthly) < LOOKBACK + 2:
        raise RuntimeError(
            f"Only {len(monthly)} monthly bars available; need at least {LOOKBACK + 2}."
        )

    # 12-1 momentum: shift(1) skips last month, pct_change(LOOKBACK) looks back 12m
    mom = monthly.shift(1).pct_change(LOOKBACK)

    # Use the most recent completed month-end row
    latest = mom.iloc[-1].dropna()
    latest = latest[latest > 0]                            # absolute momentum filter
    target = latest.nlargest(min(N_POSITIONS, len(latest))).index.tolist()

    log.info("Signal date : %s (prices through %s)",
             monthly.index[-2].strftime("%Y-%m-%d"),
             monthly.index[-1].strftime("%Y-%m-%d"))
    log.info("Target portfolio (%d stocks): %s", len(target), target)

    # Also capture the most recent daily close for every universe ticker.
    # These are used as fallback prices for position sizing (no IB subscription needed).
    yf_prices: dict[str, float] = daily.iloc[-1].dropna().to_dict()
    log.info("yfinance fallback prices loaded for %d tickers (last date: %s).",
             len(yf_prices), daily.index[-1].strftime("%Y-%m-%d"))

    return target, yf_prices


# ── FX rate ───────────────────────────────────────────────────────────────────
def get_eurusd_rate() -> float:
    """Fetch the latest EUR/USD exchange rate via yfinance."""
    data = yf.download("EURUSD=X", period="5d", auto_adjust=True, progress=False)["Close"]
    rate = float(data.dropna().iloc[-1])
    log.info("EUR/USD rate : %.4f  (1 € = $%.4f)", rate, rate)
    return rate


# ── IB helpers ────────────────────────────────────────────────────────────────
def connect_ib() -> IB:
    ib = IB()
    log.info("Connecting to IB at %s:%s (clientId=%s) ...", IB_HOST, IB_PORT, IB_CLIENT)
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT)
    log.info("Connected.  Account: %s", ib.wrapper.accounts)
    return ib


def get_account_nav(ib: IB) -> float:
    """Return total Net Asset Value (equity) of the account in its base currency."""
    account_values = ib.accountSummary()
    # IB reports NetLiquidation per currency AND a BASE summary row.
    # We prefer the BASE row; fall back to the first non-zero NetLiquidation found.
    base_nav = None
    fallback_nav = None
    for av in account_values:
        if av.tag == "NetLiquidation":
            val = float(av.value)
            if val == 0:
                continue
            if av.currency == "BASE":
                base_nav = val
                log.info("Account NAV (BASE): %.2f", val)
            elif fallback_nav is None:
                fallback_nav = val
                log.info("Account NAV (%s): %.2f", av.currency, val)
    nav = base_nav if base_nav is not None else fallback_nav
    if nav is None:
        # Debug: print all tags received so the user can diagnose
        tags = [(av.tag, av.currency, av.value) for av in account_values]
        log.error("All account summary rows: %s", tags)
        raise RuntimeError(
            "Could not retrieve NetLiquidation from IB account summary. "
            "See logged rows above for what was returned."
        )
    return nav


def get_current_positions(ib: IB) -> dict[str, float]:
    """Return {ticker: shares} for all equity positions currently held."""
    positions = {}
    for pos in ib.positions():
        if pos.contract.secType == "STK" and pos.contract.currency == "USD":
            ticker = pos.contract.symbol
            positions[ticker] = pos.position
    log.info("Current holdings (%d): %s", len(positions), list(positions.keys()))
    return positions


def get_last_prices(
    ib: IB,
    tickers: list[str],
    yf_fallback: dict[str, float],
    eurusd: float,
) -> dict[str, float]:
    """
    Fetch prices for position sizing.

    Priority:
      1. IB delayed/frozen market data  (free, no subscription required)
      2. yfinance last daily close       (already downloaded for signal computation)

    IB paper accounts have no real-time data subscription by default.
    We request delayed-frozen data (type 4) which is always available and
    returns the last known price even when the market is closed.
    """
    # Request delayed-frozen data — free for all IB accounts, works on weekends
    # Type 1 = live | 2 = frozen | 3 = delayed | 4 = delayed frozen
    ib.reqMarketDataType(4)

    contracts = [Stock(t, "SMART", "USD") for t in tickers]
    ib.qualifyContracts(*contracts)

    prices: dict[str, float] = {}
    tickers_data = ib.reqTickers(*contracts)
    for td in tickers_data:
        sym = td.contract.symbol
        # last → close → delayed last → delayed close
        px = (td.last   if (td.last  and td.last  > 0) else
              td.close  if (td.close and td.close > 0) else
              td.delayedLast  if (hasattr(td, "delayedLast")  and td.delayedLast  and td.delayedLast  > 0) else
              td.delayedClose if (hasattr(td, "delayedClose") and td.delayedClose and td.delayedClose > 0) else
              None)
        if px:
            prices[sym] = px
            log.info("IB price  %-6s  €%.2f  ($%.2f)", sym, px / eurusd, px)

    # Fill any gaps with yfinance closes
    missing = [t for t in tickers if t not in prices]
    if missing:
        log.info("Falling back to yfinance prices for: %s", missing)
        for t in missing:
            if t in yf_fallback:
                prices[t] = yf_fallback[t]
                log.info("yf price  %-6s  €%.2f  ($%.2f)", t, yf_fallback[t] / eurusd, yf_fallback[t])
            else:
                log.warning("No price found anywhere for %s — skipping.", t)

    return prices


# ── Trade planning ────────────────────────────────────────────────────────────
def compute_trades(
    target: list[str],
    current_positions: dict[str, float],
    prices: dict[str, float],
    nav: float,
    eurusd: float,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Compute shares to buy and sell to move from current_positions to an
    equal-weight target portfolio.

    Returns (buys, sells) where each is {ticker: shares}.
    Tickers without a valid price are dropped with a warning.
    """
    # Drop targets with no price data
    priceable_targets = [t for t in target if t in prices]
    if len(priceable_targets) < len(target):
        missing = set(target) - set(priceable_targets)
        log.warning("Dropping targets with no price: %s", missing)

    n = len(priceable_targets)
    if n == 0:
        log.error("No priceable targets — nothing to trade.")
        return {}, {}

    # nav is in EUR; prices are in USD — convert the per-stock allocation to USD
    target_alloc_per_stock_usd = (nav / n) * eurusd

    buys: dict[str, int]  = {}
    sells: dict[str, int] = {}

    # Sells: positions not in target or that need to shrink
    for ticker, shares in current_positions.items():
        if ticker not in priceable_targets:
            if shares > 0:
                sells[ticker] = int(shares)
        else:
            # Will be reconciled below
            pass

    # Buys / adjustments for target stocks
    for ticker in priceable_targets:
        px = prices[ticker]
        target_shares = int(target_alloc_per_stock_usd / px)  # floor to whole shares
        current_shares = int(current_positions.get(ticker, 0))
        delta = target_shares - current_shares

        if delta > 0:
            buys[ticker] = delta
        elif delta < 0:
            sells[ticker] = sells.get(ticker, 0) + abs(delta)

    return buys, sells


def print_trade_plan(
    target: list[str],
    current_positions: dict[str, float],
    prices: dict[str, float],
    buys: dict[str, int],
    sells: dict[str, int],
    nav: float,
    eurusd: float,
) -> None:
    """Pretty-print the full rebalance plan. All monetary values in EUR."""
    sep = "─" * 70
    alloc_eur = nav / max(len(target), 1)
    print()
    print("═" * 70)
    print(f"  ISM1 REBALANCE PLAN  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Account NAV : €{nav:,.2f}  (EUR/USD: {eurusd:.4f})")
    print(f"  Target      : {len(target)} stocks  (equal-weight ~€{alloc_eur:,.0f} each)")
    print("═" * 70)

    print(f"\n  {'SELLS':}")
    print(f"  {sep}")
    if sells:
        for t, qty in sorted(sells.items()):
            px_eur = prices.get(t, 0) / eurusd
            val    = qty * px_eur
            print(f"  SELL  {qty:>5} {t:<8}  @ ~€{px_eur:>8.2f}  =>  ~€{val:>10,.0f}")
    else:
        print("  (none)")

    print(f"\n  {'BUYS':}")
    print(f"  {sep}")
    if buys:
        for t, qty in sorted(buys.items()):
            px_eur = prices.get(t, 0) / eurusd
            val    = qty * px_eur
            print(f"  BUY   {qty:>5} {t:<8}  @ ~€{px_eur:>8.2f}  =>  ~€{val:>10,.0f}")
    else:
        print("  (none)")

    print(f"\n  {'HOLDS (no change)':}")
    print(f"  {sep}")
    holds = [t for t in target if t not in buys and t not in sells
             and t in current_positions]
    if holds:
        for t in sorted(holds):
            px_eur = prices.get(t, 0) / eurusd
            qty    = int(current_positions.get(t, 0))
            print(f"  HOLD  {qty:>5} {t:<8}  @ ~€{px_eur:>8.2f}")
    else:
        print("  (none)")

    total_sell_eur = sum(sells.get(t, 0) * prices.get(t, 0) / eurusd for t in sells)
    total_buy_eur  = sum(buys.get(t,  0) * prices.get(t, 0) / eurusd for t in buys)
    print(f"\n  Total sell notional : ~€{total_sell_eur:>12,.0f}")
    print(f"  Total buy  notional : ~€{total_buy_eur:>12,.0f}")
    print("═" * 70)
    print()


# ── Order execution ───────────────────────────────────────────────────────────
def execute_trades(
    ib: IB,
    buys: dict[str, int],
    sells: dict[str, int],
) -> None:
    """
    Place market orders for all sells first, then all buys.
    Waits for each batch to fill before continuing.
    """
    def place_and_wait(orders_dict: dict[str, int], action: str) -> None:
        if not orders_dict:
            return
        trades = []
        for ticker, qty in orders_dict.items():
            if qty <= 0:
                continue
            contract = Stock(ticker, "SMART", "USD")
            ib.qualifyContracts(contract)
            order = MarketOrder(action, qty)
            trade = ib.placeOrder(contract, order)
            log.info("Placed %s %d %s (orderId=%s)", action, qty, ticker, trade.order.orderId)
            trades.append(trade)

        # Wait up to 60 seconds for all orders to fill
        deadline = time.time() + 60
        while time.time() < deadline:
            ib.sleep(2)
            statuses = [t.orderStatus.status for t in trades]
            filled   = [s in ("Filled", "Cancelled", "Inactive") for s in statuses]
            log.info("%s batch status: %s", action, statuses)
            if all(filled):
                break
        else:
            log.warning("Timeout waiting for %s orders to fill. Check TWS manually.", action)

    log.info("Executing SELLS ...")
    place_and_wait(sells, "SELL")

    log.info("Executing BUYS ...")
    place_and_wait(buys, "BUY")

    log.info("All orders submitted.")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    print(__doc__)

    if DRY_RUN:
        log.info("=" * 60)
        log.info("DRY RUN MODE — no orders will be placed.")
        log.info("Set DRY_RUN = False to execute live trades.")
        log.info("=" * 60)

    # 1. Compute today's target portfolio from market data
    target, yf_prices = compute_target_portfolio()

    # 2. Fetch EUR/USD rate (NAV is in EUR, stock prices are in USD)
    eurusd = get_eurusd_rate()

    # 3. Connect to IB
    ib = connect_ib()

    try:
        # 4. Gather account state
        nav               = get_account_nav(ib)
        current_positions = get_current_positions(ib)

        # 5. Fetch prices for all tickers we care about (targets + current holdings)
        all_tickers = list(set(target) | set(current_positions.keys()))
        log.info("Fetching prices for %d tickers (IB delayed → yfinance fallback) ...",
                 len(all_tickers))
        prices = get_last_prices(ib, all_tickers, yf_prices, eurusd)

        # 6. Plan the rebalance
        buys, sells = compute_trades(target, current_positions, prices, nav, eurusd)
        print_trade_plan(target, current_positions, prices, buys, sells, nav, eurusd)

        # 6. Execute (or skip in dry-run)
        if DRY_RUN:
            log.info("DRY RUN — skipping order placement. Review the plan above.")
        else:
            confirm = input("Type YES to execute the trades above: ").strip()
            if confirm.upper() == "YES":
                execute_trades(ib, buys, sells)
            else:
                log.info("Aborted by user.")

    finally:
        ib.disconnect()
        log.info("Disconnected from IB.")


if __name__ == "__main__":
    main()
