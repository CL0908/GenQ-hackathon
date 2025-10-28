# ====== Realistic PFE for a European option (CCR): ARMA–GARCH(t), path-wise σ, optional jumps ======
# Keeps your robust_prices() loader unchanged (Yahoo→Stooq→cache).
# New bits: GARCH(1,1) w/ Student-t innovations, optional jumps, per-path repricing vol, PD/LGD ECL.

import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

def robust_prices(ticker, lookback_years):
    end = datetime.today()
    start = end - timedelta(days=int(365.25 * lookback_years))
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("Download failed.")
    return df


try:
    from arch import arch_model
except Exception:
    arch_model = None  # Fallback to your ARIMA if arch not installed

# ---------- Your existing config (tweak as you like) ----------
TICKER = "ENPH"
LOOKBACK_YEARS = 5
RISK_FREE = 0.02
OPTION_SIDE = "call"     # "call" or "put"
STRIKE = 150.0
EXPIRY = "2026-06-19"
QUANTILE = 0.95
HORIZON_DAYS = [10, 20, 60, 120]
N_SCEN = 20000

# Credit assumptions (for simple loss reporting)
PD_1Y = 0.01            # 1-year default probability of the counterparty (1%)
LGD   = 0.60            # loss-given-default (60%)

# Realism toggles
USE_JUMPS = True
LAMBDA_J  = 0.01        # daily jump probability (1%)
MU_JUMP   = 0.00        # average jump size in daily return space
SIG_JUMP  = 0.05        # jump volatility (5% daily)

# Cache file (unchanged)
CACHE_CSV = f"{TICKER}_adjclose_cache.csv"

# ---------- Black–Scholes ----------
def bs_price(spot, K, r, tau, vol, side="call"):
    if tau <= 0 or vol <= 0:
        intrinsic = max(spot - K, 0.0) if side == "call" else max(K - spot, 0.0)
        return intrinsic
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(spot / K) + (r + 0.5 * vol**2) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau
    if side.lower() == "call":
        return spot * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    else:
        return K * np.exp(-r * tau) * norm.cdf(-d2) - spot * norm.cdf(-d1)

# ---------- MAIN ----------
def main():
    # 1) Load prices (uses your robust_prices)
    px = robust_prices(TICKER, LOOKBACK_YEARS)
    if "Close" not in px.columns:
        raise RuntimeError("Price frame missing 'Close' column.")
    close = px["Close"].dropna()
    if len(close) < 260:
        raise RuntimeError(f"Not enough data for modelling (got {len(close)} days).")

    S0 = float(close.iloc[-1])

    # 2) Build the return series (daily)
    # We'll use arithmetic returns for GARCH; accumulate via log(1+r) to form S_t
    r = close.pct_change().dropna()
    if r.std() == 0 or np.isnan(r.std()):
        raise RuntimeError("Return series not usable (zero or NaN volatility).")

    # 3) Fit ARMA(1,0)-GARCH(1,1) with Student-t innovations (fat tails)
    # arch typically expects percent returns; scale by 100 for better numerics
    use_garch = arch_model is not None
    if use_garch:
        am = arch_model(r * 100.0, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t")
        res = am.fit(disp="off")
        # Long-run variance etc. are embedded in params; we will simulate forward
    else:
        # Fallback: ARIMA(1,0,1) on log-returns (less realistic)
        lr = np.log(close).diff().dropna()
        arma_fit = ARIMA(lr, order=(1,0,1)).fit()

    # Helper: PD over horizon h (days) from 1Y PD (simple no-seasonal scaling)
    def horizon_pd(pd_1y, h_days):
        return 1.0 - (1.0 - pd_1y) ** (h_days / 252.0)

    # 4) Simulate price & per-path repricing volatility at horizon
    rng = np.random.default_rng(42)

    def simulate_paths_garch(h_days, reps):
        """
        Returns:
          Sh: (reps,) simulated spot at horizon
          sig_ann_h: (reps,) annualized conditional vol at horizon (from GARCH)
        """
        sim = res.simulate(res.params, nobs=h_days, repetitions=reps)
        # sim['data'] are simulated returns in percent; convert to decimal
        ret_path = np.asarray(sim["data"]) / 100.0               # shape (h_days, reps)
        sig_daily = np.asarray(sim["volatility"]) / 100.0        # daily σ per step (h_days, reps)
        # Optional jumps
        if USE_JUMPS:
            # Bernoulli(lambda) * Normal(MU_JUMP, SIG_JUMP)
            jumps = rng.normal(MU_JUMP, SIG_JUMP, size=ret_path.shape) * (rng.random(ret_path.shape) < LAMBDA_J)
            ret_path = ret_path + jumps
        # Accumulate to price via log(1+r)
        cum_log = np.log1p(ret_path).sum(axis=0)                 # (reps,)
        Sh = S0 * np.exp(cum_log)
        # Take the last day’s conditional vol as repricing σ at horizon
        sig_daily_h = sig_daily[-1, :]                           # (reps,)
        sig_ann_h = sig_daily_h * np.sqrt(252.0)
        return Sh, sig_ann_h

    def simulate_paths_arima(h_days, reps):
        # Fallback if arch not installed: simulate log-returns with ARIMA and constant σ
        sims = arma_fit.simulate(nsimulations=h_days, repetitions=reps)
        sims = np.asarray(sims)                                  # log-returns
        cum_log = sims.sum(axis=0)
        Sh = S0 * np.exp(cum_log)
        # Use historical ann vol as repricing σ for everyone (less realistic)
        ann_vol_hist = (np.log(close).diff().dropna().std()) * np.sqrt(252.0)
        sig_ann_h = np.full(reps, float(ann_vol_hist))
        return Sh, sig_ann_h

    # 5) Valuation & exposure metrics
    def compute_pfe_for_horizon(h_days):
        if use_garch:
            Sh, sig_ann = simulate_paths_garch(h_days, N_SCEN)
        else:
            Sh, sig_ann = simulate_paths_arima(h_days, N_SCEN)

        # Time remaining to expiry (years) when we arrive at horizon h
        today = close.index[-1].to_pydatetime()
        expiry_dt = datetime.strptime(EXPIRY, "%Y-%m-%d")
        T0 = max((expiry_dt - today).days, 0) / 365.0
        tau = max(T0 - h_days / 252.0, 0.0)

        # Path-wise valuation with each path’s σ
        if OPTION_SIDE.lower() == "call":
            vals = np.array([bs_price(s, STRIKE, RISK_FREE, tau, sig, "call") for s, sig in zip(Sh, sig_ann)])
        else:
            vals = np.array([bs_price(s, STRIKE, RISK_FREE, tau, sig, "put") for s, sig in zip(Sh, sig_ann)])

        # Long position exposure (MtM ≥ 0); flip sign if you want short exposure
        exposure = np.maximum(vals, 0.0)

        # PFE & EE
        pfe_q = float(np.quantile(exposure, QUANTILE))
        ee    = float(np.mean(exposure))

        # Simple credit loss view at horizon (not portfolio aggregated):
        pd_h  = horizon_pd(PD_1Y, h_days)
        ecl   = ee * LGD * pd_h

        return {
            "h_days": h_days,
            "tau_remaining_yrs": tau,
            "PFE": pfe_q,
            "EE": ee,
            "ECL": float(ecl),
            "pd_h": float(pd_h),
            "spot_now": S0,
            "spot_median_h": float(np.median(Sh)),
            "repricing_vol_ann_median": float(np.median(sig_ann)),
            "repricing_vol_ann_p90": float(np.quantile(sig_ann, 0.90)),
            "repricing_vol_ann_p99": float(np.quantile(sig_ann, 0.99)),
            "engine": "AR(1)-GARCH(1,1)-t" if use_garch else "ARIMA(1,0,1)"
        }

    # 6) Run & print
    header = {
        "ticker": TICKER,
        "last_spot": S0,
        "option": {"side": OPTION_SIDE, "K": STRIKE, "expiry": EXPIRY},
        "risk_free": RISK_FREE,
        "quantile": QUANTILE,
        "simulations_per_horizon": N_SCEN,
        "engine": "AR(1)-GARCH(1,1)-t" if use_garch else "ARIMA(1,0,1)",
        "jumps": {"enabled": USE_JUMPS, "lambda": LAMBDA_J, "mu": MU_JUMP, "sigma": SIG_JUMP},
        "credit": {"PD_1Y": PD_1Y, "LGD": LGD}
    }
    print(json.dumps(header, indent=2))
    print("---- Results ----")
    for h in HORIZON_DAYS:
        out = compute_pfe_for_horizon(h)
        print(
            f"h={out['h_days']:>4}d | tau_rem≈{out['tau_remaining_yrs']:.3f}y | "
            f"PFE({int(QUANTILE*100)}%)={out['PFE']:.6f} | EE={out['EE']:.6f} | "
            f"ECL={out['ECL']:.8f} | PD_h={out['pd_h']:.4%} | "
            f"median S_h={out['spot_median_h']:.4f} | "
            f"σ_ann@h (med/p90/p99)=({out['repricing_vol_ann_median']:.4f}/"
            f"{out['repricing_vol_ann_p90']:.4f}/{out['repricing_vol_ann_p99']:.4f}) | "
            f"engine={out['engine']}"
        )

if __name__ == "__main__":
    main()
