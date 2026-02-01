import os
import json
import joblib
import requests
import yfinance as yf
import pandas as pd
import numpy as np


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


def make_features(dfw: pd.DataFrame) -> pd.DataFrame:
    df = dfw.copy()
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
    return df.dropna()


def decision_from_proba(p_up: float) -> str:
    if p_up > 0.75:
        return "BUY WEEK ✅"
    elif p_up < 0.25:
        return "SELL WEEK ✅"
    else:
        return "NEUTRAL / NO-TRADE ⚠️"


def pct_change(a, b) -> float:
    return (a / b - 1.0) * 100.0


def main():
    model = joblib.load("weekly_bias_model.joblib")
    with open("feature_cols.json", "r") as f:
        feature_cols = json.load(f)

    dfw = build_weekly_dataset()
    feat = make_features(dfw)

    X_last = feat.tail(1).reindex(columns=feature_cols)

    if hasattr(model, "predict_proba"):
        p_up = float(model.predict_proba(X_last)[0, 1])
    else:
        pred = int(model.predict(X_last)[0])
        p_up = 0.55 if pred == 1 else 0.45

    print("\n=================================")
    print("WEEKLY BIAS PREDICTION (ULTIMATE FINAL)")
    print("=================================")
    print("Last weekly date:", X_last.index[0].date())
    print("P(Next week UP):", round(p_up, 3))
    print("Decision:", decision_from_proba(p_up))

    last2 = dfw.tail(2)
    if len(last2) == 2:
        print("\n--- Drivers (last week change) ---")
        print("Δ Gold (%):", round(pct_change(last2['gold'].iloc[-1], last2['gold'].iloc[-2]), 3))
        print("Δ DXY (%):", round(pct_change(last2['dxy'].iloc[-1], last2['dxy'].iloc[-2]), 3))
        print("Δ VIX (%):", round(pct_change(last2['vix'].iloc[-1], last2['vix'].iloc[-2]), 3))
        print("Δ SPX (%):", round(pct_change(last2['spx'].iloc[-1], last2['spx'].iloc[-2]), 3))
        print("Δ TNX (%):", round(pct_change(last2['tnx'].iloc[-1], last2['tnx'].iloc[-2]), 3))
        print("Δ IEF (%):", round(pct_change(last2['ief'].iloc[-1], last2['ief'].iloc[-2]), 3))
        print("Δ USO (%):", round(pct_change(last2['uso'].iloc[-1], last2['uso'].iloc[-2]), 3))
        print("Δ Real10 (points):", round(float(last2["real10"].iloc[-1] - last2["real10"].iloc[-2]), 4))
        print("Δ Breakeven (points):", round(float(last2["breakeven"].iloc[-1] - last2["breakeven"].iloc[-2]), 4))


if __name__ == "__main__":
    main()
