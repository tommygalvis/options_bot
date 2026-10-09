"""
Options Trading Bot
===================
Covers:
  - Black-Scholes-Merton pricing (calls & puts)
  - Greeks (Delta, Gamma, Theta, Vega, Rho)
  - Implied Volatility solver (Newton-Raphson + Brent fallback)
  - Historical & rolling volatility
  - Volatility surface interpolation
  - Delta hedging engine
  - Position & risk management (VaR, CVaR, Kelly)
  - Basic signal generation (IV vs RV spread)

Dependencies: numpy, scipy, pandas
Install:  pip install numpy scipy pandas
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Literal, Optional
from dataclasses import dataclass, field
from datetime import datetime


# ─────────────────────────────────────────────
# 1. BLACK-SCHOLES-MERTON CORE
# ─────────────────────────────────────────────

def bsm_d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    """Compute d1 and d2 for Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Black-Scholes-Merton option price.

    Parameters
    ----------
    S     : Current underlying price
    K     : Strike price
    T     : Time to expiry in years
    r     : Risk-free rate (annualised, continuous)
    sigma : Implied/historical volatility (annualised)
    option_type : 'call' or 'put'

    Returns
    -------
    Option fair value
    """
    d1, d2 = bsm_d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def put_call_parity_check(
    call_price: float, put_price: float, S: float, K: float, T: float, r: float
) -> float:
    """
    Returns the put-call parity deviation (should be ~0 for no-arb).
    C - P = S - K * e^(-rT)
    """
    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)
    return lhs - rhs


# ─────────────────────────────────────────────
# 2. THE GREEKS
# ─────────────────────────────────────────────

def delta(S, K, T, r, sigma, option_type="call") -> float:
    """dV/dS — sensitivity of price to underlying move."""
    d1, _ = bsm_d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma) -> float:
    """d²V/dS² — rate of change of delta (same for calls and puts)."""
    d1, _ = bsm_d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(S, K, T, r, sigma, option_type="call") -> float:
    """dV/dT — daily time decay (returned as per-day figure)."""
    d1, d2 = bsm_d1_d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        t = term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        t = term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return t / 365  # convert annual to daily


def vega(S, K, T, r, sigma) -> float:
    """dV/dσ — sensitivity to 1% move in vol (same for calls and puts)."""
    d1, _ = bsm_d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) * 0.01  # per 1% vol move


def rho(S, K, T, r, sigma, option_type="call") -> float:
    """dV/dr — sensitivity to interest rate."""
    _, d2 = bsm_d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01


def all_greeks(S, K, T, r, sigma, option_type="call") -> dict:
    """Return all Greeks in a single dict."""
    return {
        "price":  bsm_price(S, K, T, r, sigma, option_type),
        "delta":  delta(S, K, T, r, sigma, option_type),
        "gamma":  gamma(S, K, T, r, sigma),
        "theta":  theta(S, K, T, r, sigma, option_type),
        "vega":   vega(S, K, T, r, sigma),
        "rho":    rho(S, K, T, r, sigma, option_type),
    }


# ─────────────────────────────────────────────
# 3. IMPLIED VOLATILITY SOLVER
# ─────────────────────────────────────────────

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"] = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Compute implied volatility via Newton-Raphson with Brent fallback.

    Returns
    -------
    Implied volatility as a decimal (e.g. 0.25 = 25%)
    """
    # Intrinsic value check
    intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
    if market_price < intrinsic:
        raise ValueError("Market price is below intrinsic value — no valid IV.")

    # Newton-Raphson
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        price = bsm_price(S, K, T, r, sigma, option_type)
        v = vega(S, K, T, r, sigma) / 0.01  # undo the 1% scaling for full vega
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        if abs(v) < 1e-10:
            break  # vega too small, fall back to Brent
        sigma -= diff / v
        if sigma <= 0:
            sigma = 1e-4

    # Brent's method fallback (robust bracket search)
    try:
        iv = brentq(
            lambda s: bsm_price(S, K, T, r, s, option_type) - market_price,
            1e-4,
            5.0,
            xtol=tol,
        )
        return iv
    except ValueError:
        raise ValueError("Could not find a valid implied volatility.")


# ─────────────────────────────────────────────
# 4. VOLATILITY CALCULATIONS
# ─────────────────────────────────────────────

def historical_vol(prices: pd.Series, window: int = 30, trading_days: int = 252) -> pd.Series:
    """
    Rolling historical (realised) volatility from a price series.

    Parameters
    ----------
    prices       : Daily closing prices
    window       : Rolling window in days
    trading_days : Annualisation factor

    Returns
    -------
    Annualised rolling volatility series
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(trading_days)


def vol_cone(prices: pd.Series, windows=(10, 20, 30, 60, 90)) -> pd.DataFrame:
    """
    Build a vol cone — percentile table of realised vol across windows.
    Useful for comparing current IV against historical distribution.
    """
    rows = {}
    for w in windows:
        rv = historical_vol(prices, window=w).dropna()
        rows[w] = {
            "min":    rv.min(),
            "10%":    rv.quantile(0.10),
            "25%":    rv.quantile(0.25),
            "median": rv.median(),
            "75%":    rv.quantile(0.75),
            "90%":    rv.quantile(0.90),
            "max":    rv.max(),
            "current": rv.iloc[-1],
        }
    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────
# 5. VOLATILITY SURFACE
# ─────────────────────────────────────────────

class VolSurface:
    """
    Simple bilinear-interpolated volatility surface over
    (strike, expiry) space.

    Usage
    -----
    surface = VolSurface(strikes, expiries, iv_matrix)
    iv = surface.get_iv(strike=105, expiry=0.25)
    """

    def __init__(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        iv_matrix: np.ndarray,
    ):
        """
        Parameters
        ----------
        strikes   : Array of strike prices (shape: N)
        expiries  : Array of expiry times in years (shape: M)
        iv_matrix : IV grid, shape (N, M) — iv_matrix[i, j] is IV
                    for strikes[i] and expiries[j]
        """
        self.strikes = np.array(strikes)
        self.expiries = np.array(expiries)
        self.iv_matrix = np.array(iv_matrix)

    def get_iv(self, strike: float, expiry: float) -> float:
        """Bilinear interpolation of IV for any (strike, expiry) point."""
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (self.strikes, self.expiries),
            self.iv_matrix,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        return float(interp([[strike, expiry]])[0])

    def atm_term_structure(self, S: float) -> pd.Series:
        """Return ATM IV across all expiries."""
        ivs = [self.get_iv(S, t) for t in self.expiries]
        return pd.Series(ivs, index=self.expiries, name="ATM IV")

    def skew(self, expiry: float) -> pd.Series:
        """Return the vol smile/skew at a given expiry."""
        ivs = [self.get_iv(k, expiry) for k in self.strikes]
        return pd.Series(ivs, index=self.strikes, name=f"Skew T={expiry:.2f}")


# ─────────────────────────────────────────────
# 6. POSITION & RISK MANAGEMENT
# ─────────────────────────────────────────────

@dataclass
class Position:
    """Single options position."""
    symbol: str
    option_type: Literal["call", "put"]
    strike: float
    expiry: float          # years to expiry
    quantity: int          # positive = long, negative = short
    entry_price: float
    S: float               # underlying price at entry
    r: float = 0.05
    sigma: float = 0.20
    timestamp: datetime = field(default_factory=datetime.now)

    def mark_to_market(self, S: float, sigma: float, r: float = None) -> float:
        r = r or self.r
        price = bsm_price(S, self.strike, self.expiry, r, sigma, self.option_type)
        return price * self.quantity * 100  # standard lot = 100 shares

    def greeks_now(self, S: float, sigma: float, r: float = None) -> dict:
        r = r or self.r
        g = all_greeks(S, self.strike, self.expiry, r, sigma, self.option_type)
        # Scale by position size (100 shares per contract)
        return {k: v * self.quantity * 100 for k, v in g.items()}


@dataclass
class Portfolio:
    """Collection of option positions with aggregate risk."""
    positions: list[Position] = field(default_factory=list)

    def add(self, position: Position):
        self.positions.append(position)

    def aggregate_greeks(self, S: float, sigma: float, r: float = 0.05) -> dict:
        """Sum Greeks across all positions."""
        totals = {"price": 0, "delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
        for pos in self.positions:
            g = pos.greeks_now(S, sigma, r)
            for k in totals:
                totals[k] += g[k]
        return totals

    def net_delta(self, S: float, sigma: float, r: float = 0.05) -> float:
        return self.aggregate_greeks(S, sigma, r)["delta"]


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Value at Risk.
    Returns the loss not exceeded at the given confidence level.
    """
    return -np.percentile(returns, (1 - confidence) * 100)


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall) — average loss beyond VaR.
    """
    var = historical_var(returns, confidence)
    tail = returns[returns < -var]
    return -tail.mean() if not tail.empty else var


def kelly_fraction(win_prob: float, win_return: float, loss_return: float) -> float:
    """
    Kelly Criterion for position sizing.
    f* = (b*p - q) / b
    where b = win_return/loss_return, p = win_prob, q = 1 - p

    Returns fraction of capital to risk (cap at 25% for safety).
    """
    q = 1 - win_prob
    b = abs(win_return / loss_return)
    f = (b * win_prob - q) / b
    return max(0.0, min(f, 0.25))  # never risk more than 25%


# ─────────────────────────────────────────────
# 7. DELTA HEDGING ENGINE
# ─────────────────────────────────────────────

class DeltaHedger:
    """
    Simulates a delta-neutral hedging strategy over a price path.

    The hedger holds an option position and rebalances a stock position
    at each step to maintain net delta ≈ 0.
    """

    def __init__(
        self,
        option_type: Literal["call", "put"],
        K: float,
        T: float,
        r: float,
        sigma: float,
        quantity: int = 1,
        rebalance_threshold: float = 0.05,
    ):
        self.option_type = option_type
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.quantity = quantity
        self.threshold = rebalance_threshold
        self.stock_position = 0.0
        self.cash = 0.0
        self.hedge_log = []

    def _time_remaining(self, step: int, total_steps: int) -> float:
        return max(self.T * (1 - step / total_steps), 1e-6)

    def run(self, price_path: np.ndarray) -> pd.DataFrame:
        """
        Run the hedge simulation over a price path.

        Parameters
        ----------
        price_path : Array of underlying prices (e.g. daily closes)

        Returns
        -------
        DataFrame with hedge log: price, delta, stock_position, pnl
        """
        self.hedge_log = []
        total_steps = len(price_path) - 1
        self.stock_position = 0.0
        self.cash = 0.0

        for i, S in enumerate(price_path):
            t = self._time_remaining(i, total_steps)
            d = delta(S, self.K, t, self.r, self.sigma, self.option_type)
            option_delta = d * self.quantity * 100

            # Rebalance if delta shift exceeds threshold
            hedge_needed = -option_delta
            diff = hedge_needed - self.stock_position
            if abs(diff) > self.threshold * 100 or i == 0:
                self.cash -= diff * S  # buy/sell stock at current price
                self.stock_position = hedge_needed

            option_price = bsm_price(S, self.K, t, self.r, self.sigma, self.option_type)
            portfolio_value = (
                option_price * self.quantity * 100
                + self.stock_position * S
                + self.cash
            )

            self.hedge_log.append({
                "step": i,
                "S": S,
                "T_remaining": t,
                "option_delta": option_delta,
                "stock_position": self.stock_position,
                "option_price": option_price,
                "portfolio_value": portfolio_value,
                "cash": self.cash,
            })

        return pd.DataFrame(self.hedge_log)


# ─────────────────────────────────────────────
# 8. SIGNAL GENERATION (IV vs RV Spread)
# ─────────────────────────────────────────────

def iv_rv_signal(
    implied_vol: float,
    realized_vol: float,
    threshold: float = 0.02,
) -> str:
    """
    Basic vol arbitrage signal.

    If IV >> RV  → options are expensive → consider selling vol (short straddle/strangle)
    If IV << RV  → options are cheap     → consider buying vol  (long straddle/strangle)

    Parameters
    ----------
    implied_vol   : Current IV (e.g. from vol surface)
    realized_vol  : Recent realised vol (e.g. 30-day HV)
    threshold     : Minimum spread to generate a signal

    Returns
    -------
    'sell_vol', 'buy_vol', or 'neutral'
    """
    spread = implied_vol - realized_vol
    if spread > threshold:
        return "sell_vol"
    elif spread < -threshold:
        return "buy_vol"
    return "neutral"


def forward_vol(sigma1: float, T1: float, sigma2: float, T2: float) -> float:
    """
    Implied forward volatility between T1 and T2.
    σ²_fwd = (σ²₂·T₂ - σ²₁·T₁) / (T₂ - T₁)
    """
    if T2 <= T1:
        raise ValueError("T2 must be greater than T1")
    var_fwd = (sigma2 ** 2 * T2 - sigma1 ** 2 * T1) / (T2 - T1)
    if var_fwd < 0:
        raise ValueError("Negative forward variance — check inputs.")
    return np.sqrt(var_fwd)


# ─────────────────────────────────────────────
# 9. MONTE CARLO PRICER
# ─────────────────────────────────────────────

def monte_carlo_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    n_paths: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo option pricer with confidence interval.

    Uses risk-neutral GBM:  S_T = S * exp((r - σ²/2)*T + σ*√T*Z)

    Returns
    -------
    dict with: price, stderr, ci_lower, ci_upper
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    stderr = discounted.std() / np.sqrt(n_paths)

    return {
        "price":    price,
        "stderr":   stderr,
        "ci_lower": price - 1.96 * stderr,
        "ci_upper": price + 1.96 * stderr,
    }


# ─────────────────────────────────────────────
# 10. DEMO / MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("OPTIONS TRADING BOT — DEMO")
    print("=" * 60)

    # --- Parameters ---
    S, K, T, r, sigma = 100, 105, 0.5, 0.05, 0.20

    # 1. BSM Pricing
    print("\n[1] BSM PRICING")
    call = bsm_price(S, K, T, r, sigma, "call")
    put  = bsm_price(S, K, T, r, sigma, "put")
    print(f"  Call price : ${call:.4f}")
    print(f"  Put  price : ${put:.4f}")
    print(f"  Put-call parity error: {put_call_parity_check(call, put, S, K, T, r):.6f}")

    # 2. Greeks
    print("\n[2] GREEKS (call)")
    g = all_greeks(S, K, T, r, sigma, "call")
    for name, val in g.items():
        print(f"  {name:6s}: {val:.6f}")

    # 3. Implied Volatility
    print("\n[3] IMPLIED VOLATILITY")
    iv = implied_vol(call, S, K, T, r, "call")
    print(f"  Recovered IV from BSM price: {iv:.4%}  (input was {sigma:.4%})")

    # 4. Monte Carlo
    print("\n[4] MONTE CARLO PRICER (100k paths)")
    mc = monte_carlo_price(S, K, T, r, sigma, "call")
    print(f"  MC price : ${mc['price']:.4f}  ±{mc['stderr']:.4f}")
    print(f"  95% CI   : [{mc['ci_lower']:.4f}, {mc['ci_upper']:.4f}]")
    print(f"  BSM price: ${call:.4f}")

    # 5. Historical Vol & Signals
    print("\n[5] HISTORICAL VOL & SIGNAL")
    np.random.seed(1)
    fake_prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.01, 252)))
    hv = historical_vol(fake_prices, window=30).iloc[-1]
    print(f"  30-day realised vol : {hv:.4%}")
    print(f"  Implied vol         : {sigma:.4%}")
    signal = iv_rv_signal(sigma, hv)
    print(f"  Signal              : {signal}")

    # 6. Forward Vol
    print("\n[6] FORWARD VOL (1m → 3m)")
    fv = forward_vol(sigma1=0.18, T1=1/12, sigma2=0.22, T2=3/12)
    print(f"  Forward vol (1m→3m): {fv:.4%}")

    # 7. Kelly Criterion
    print("\n[7] KELLY CRITERION")
    f = kelly_fraction(win_prob=0.55, win_return=0.10, loss_return=0.05)
    print(f"  Kelly fraction: {f:.2%} of capital")

    # 8. Delta Hedge Simulation
    print("\n[8] DELTA HEDGING SIMULATION (50-step path)")
    np.random.seed(42)
    path = 100 * np.cumprod(1 + np.random.normal(0, 0.01, 50))
    hedger = DeltaHedger("call", K=105, T=0.5, r=0.05, sigma=0.20, quantity=1)
    hedge_df = hedger.run(path)
    print(hedge_df[["step", "S", "option_delta", "stock_position", "portfolio_value"]].tail())

    # 9. Vol Surface
    print("\n[9] VOL SURFACE — ATM TERM STRUCTURE")
    strikes  = np.array([90, 95, 100, 105, 110])
    expiries = np.array([1/12, 3/12, 6/12, 1.0])
    # Realistic skewed surface (higher vol for lower strikes, term structure upward sloping)
    iv_grid  = np.array([
        [0.28, 0.27, 0.26, 0.25],
        [0.24, 0.23, 0.22, 0.22],
        [0.20, 0.20, 0.21, 0.21],
        [0.19, 0.19, 0.20, 0.21],
        [0.18, 0.19, 0.20, 0.21],
    ])
    surface = VolSurface(strikes, expiries, iv_grid)
    atm = surface.atm_term_structure(S=100)
    print(atm.to_string())

    print("\n" + "=" * 60)
    print("Done. Integrate broker API and live data feeds to go live.")
    print("=" * 60)
