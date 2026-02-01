import os
import json
import joblib
import requests
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


# =========================
# FRED API
# =========================
def fred(series_id: str) -> pd.Series:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError('❌ FRED_API_KEY vide. PowerShell: $env:FRED_API_KEY="TA_CLE"')

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        print("DEBUG URL:", r.url)
        print("DEBUG RESPONSE:", r.text[:400])
    r.raise_for_status()

    obs = r.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("date")["value"]
    s.name = series_id
    return s


# =========================
# Yahoo helpers
# =========================
def yahoo_close(ticker: str, period="20y", interval="1d") -> pd.Series:
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data is None or data.empty or "Close" not in data:
        raise ValueError(f"Yahoo data empty for {ticker}")
    s = data["Close"].copy()
    s.index = pd.to_datetime(s.index)
    s.name = ticker
    return s

def yahoo_try(tickers, period="20y", interval="1d") -> pd.Series:
    last_err = None
    for t in tickers:
        try:
            s = yahoo_close(t, period=period, interval=interval)
            print(f"✅ Using Yahoo ticker: {t}")
            return s
        except Exception as e:
            last_err = e
            print(f"⚠️ Failed Yahoo ticker: {t} | {str(e)[:140]}")
    raise last_err


# =========================
# Build weekly dataset
# =========================
def build_weekly_dataset() -> pd.DataFrame:
    gold = yahoo_try(["XAUUSD=X", "XAU=X", "GC=F"], period="20y", interval="1d"); gold.name = "gold"
    dxy  = yahoo_try(["DX-Y.NYB"], period="20y", interval="1d"); dxy.name = "dxy"
    vix  = yahoo_try(["^VIX"], period="20y", interval="1d"); vix.name = "vix"
    spx  = yahoo_try(["^GSPC"], period="20y", interval="1d"); spx.name = "spx"
    tnx  = yahoo_try(["^TNX"], period="20y", interval="1d"); tnx.name = "tnx"
    ief  = yahoo_try(["IEF"], period="20y", interval="1d"); ief.name = "ief"
    uso  = yahoo_try(["USO"], period="20y", interval="1d"); uso.name = "uso"

    real10 = fred("DFII10"); real10.name = "real10"
    breakeven = fred("T10YIE"); breakeven.name = "breakeven"

    df = pd.concat([gold, dxy, real10, breakeven, vix, spx, tnx, ief, uso], axis=1)
    dfw = df.resample("W-FRI").last().dropna()
    dfw.columns = ["gold", "dxy", "real10", "breakeven", "vix", "spx", "tnx", "ief", "uso"]
    return dfw


# =========================
# Features + Target
# =========================
def make_features(dfw: pd.DataFrame) -> pd.DataFrame:
    df = dfw.copy()

    df["gold_price"] = df["gold"]
    df["gold_ret"] = np.log(df["gold"]).diff()

    for col in ["dxy", "real10", "breakeven", "vix", "spx", "tnx", "ief", "uso"]:
        df[f"d_{col}"] = df[col].diff()

    base_cols = [
        "gold_ret",
        "d_dxy", "d_real10", "d_breakeven", "d_vix", "d_spx", "d_tnx", "d_ief", "d_uso"
    ]
    for k in [1, 2, 3]:
        for col in base_cols:
            df[f"{col}_l{k}"] = df[col].shift(k)

    df["gold_mom2"] = df["gold_ret"].rolling(2).sum()
    df["gold_vol4"] = df["gold_ret"].rolling(4).std()

    df["y"] = (df["gold_ret"].shift(-1) > 0).astype(int)
    return df.dropna()


# =========================
# Metrics
# =========================
def metrics_from_returns(r: pd.Series, periods_per_year=52) -> dict:
    r = r.dropna()
    if len(r) < 10:
        return {"total_return": None, "cagr": None, "sharpe": None, "profit_factor": None}

    equity = (1 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1)

    years = len(r) / periods_per_year
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else None

    mu = r.mean() * periods_per_year
    sig = r.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = float(mu / sig) if sig and sig != 0 else None

    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    profit_factor = float(gains / losses) if losses and losses != 0 else None

    return {"total_return": total_return, "cagr": cagr, "sharpe": sharpe, "profit_factor": profit_factor}

def max_drawdown(r: pd.Series) -> float:
    equity = (1 + r.fillna(0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


# =========================
# Backtest ULTIMATE (stable sizing)
# =========================
def backtest_weekly_ultimate(
    feat: pd.DataFrame,
    proba_up: np.ndarray,
    buy_th=0.75,
    sell_th=0.25,
    ma_window=20,
    target_vol=0.007,   # 0.7% weekly risk target
    size_cap=1.5,       # max leverage cap
    loss_cap=-0.02      # cap worst weekly loss to -2%
) -> dict:
    df = feat.copy()
    df["p_up"] = proba_up

    df["r_next"] = df["gold_ret"].shift(-1)
    df = df.dropna(subset=["r_next"])

    df["ma"] = df["gold_price"].rolling(ma_window).mean()
    df["trend_up"] = df["gold_price"] > df["ma"]

    vol = df["gold_vol4"].replace(0, np.nan)
    df["size"] = (target_vol / vol).clip(lower=0.0, upper=size_cap).fillna(0.0)

    long_sig = (df["p_up"] > buy_th) & (df["trend_up"])
    short_sig = (df["p_up"] < sell_th) & (~df["trend_up"])

    df["pos"] = 0
    df.loc[long_sig, "pos"] = 1
    df.loc[short_sig, "pos"] = -1

    df["strat_ret"] = df["pos"] * df["size"] * df["r_next"]
    df["strat_ret"] = df["strat_ret"].clip(lower=loss_cap)  # loss cap

    trades = int((df["pos"] != 0).sum())
    hit = (df.loc[df["pos"] != 0, "strat_ret"] > 0).mean() if trades > 0 else np.nan

    m = metrics_from_returns(df["strat_ret"])
    mdd = max_drawdown(df["strat_ret"])

    return {
        "trades": trades,
        "hit_rate": (float(hit) if hit == hit else None),
        "total_return": m["total_return"],
        "cagr": m["cagr"],
        "sharpe": m["sharpe"],
        "profit_factor": m["profit_factor"],
        "max_drawdown": mdd
    }


# =========================
# Train
# =========================
def main():
    dfw = build_weekly_dataset()
    feat = make_features(dfw)

    X = feat.drop(columns=["y"])
    y = feat["y"]

    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42
        )
        model_name = "XGBoost"
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(random_state=42)
        model_name = "GradientBoosting (fallback)"

    tscv = TimeSeriesSplit(n_splits=5)
    acc_scores = []
    proba_oos = pd.Series(index=X.index, dtype=float)

    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        acc_scores.append(accuracy_score(y.iloc[te], pred))

        if hasattr(model, "predict_proba"):
            proba_oos.iloc[te] = model.predict_proba(X.iloc[te])[:, 1]
        else:
            proba_oos.iloc[te] = np.where(pred == 1, 0.55, 0.45)

    model.fit(X, y)

    print("\n=================================")
    print("TRAIN WEEKLY BIAS MODEL (ULTIMATE FINAL)")
    print("=================================")
    print("Model:", model_name)
    print("Weeks used:", len(X))
    print("Walk-forward accuracy:", round(float(np.mean(acc_scores)), 3))
    print("Last weekly date:", X.index[-1].date())

    bt = backtest_weekly_ultimate(
        feat=feat,
        proba_up=proba_oos.values,
        buy_th=0.75,
        sell_th=0.25,
        ma_window=20,
        target_vol=0.007,
        size_cap=1.5,
        loss_cap=-0.02
    )

    print("\n--- Backtest ULTIMATE (MA20 + TargetVol + Cap1.5 + LossCap-2%) ---")
    print("Trades:", bt["trades"])
    print("Hit rate:", None if bt["hit_rate"] is None else round(bt["hit_rate"], 3))
    print("Total return:", None if bt["total_return"] is None else round(bt["total_return"], 3))
    print("CAGR:", None if bt["cagr"] is None else round(bt["cagr"], 3))
    print("Sharpe:", None if bt["sharpe"] is None else round(bt["sharpe"], 3))
    print("Profit factor:", None if bt["profit_factor"] is None else round(bt["profit_factor"], 3))
    print("Max drawdown:", round(bt["max_drawdown"], 3))

    joblib.dump(model, "weekly_bias_model.joblib")
    with open("feature_cols.json", "w") as f:
        json.dump(list(X.columns), f)

    print("\n✅ Saved: weekly_bias_model.joblib + feature_cols.json")


if __name__ == "__main__":
    main()
