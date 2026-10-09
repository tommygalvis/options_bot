# Options Trading Toolkit

A single-file toolkit for pricing options, measuring risk and testing simple volatility strategies. It doesn't connect to a broker or place trades.

## What's included

| Area | Functions |
|---|---|
| Pricing | `bsm_price` (Black-Scholes-Merton), `monte_carlo_price` (with confidence interval), `put_call_parity_check` |
| Greeks | `delta`, `gamma`, `theta` (per day), `vega` and `rho` (per 1% move), `all_greeks` |
| Implied volatility | `implied_vol`: Newton-Raphson with a Brent's method fallback |
| Realized volatility | `historical_vol` (rolling), `vol_cone` (percentiles across windows) |
| Volatility surface | `VolSurface`: interpolation over strike and expiry, ATM term structure, skew |
| Risk | `Portfolio` aggregate Greeks, `historical_var`, `historical_cvar`, `kelly_fraction` (capped at 25%) |
| Hedging | `DeltaHedger`: simulates delta hedging an option over a price path |
| Signals | `iv_rv_signal` (implied vs realized vol), `forward_vol` |

## Usage

```bash
pip install -r requirements.txt
python options_bot.py
```

Running the file prints a demo of each section. You can also import the functions:

```python
from options_bot import all_greeks, implied_vol

all_greeks(S=100, K=105, T=0.5, r=0.05, sigma=0.20, option_type="call")
```
