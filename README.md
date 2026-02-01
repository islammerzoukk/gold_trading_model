# Gold Trading Weekly Bias Model

This project builds a **weekly directional bias model for Gold (XAU/USD)** using:

- Macroeconomic data (Real Yields, Inflation Expectations) from FRED
- Market data (DXY, VIX, SP500, TNX, Bonds, Oil) from Yahoo Finance
- Machine Learning (XGBoost / Gradient Boosting)

The goal of this model is to give me the direction of the market gold every week then i take trades only with that direction

---

## How it works

Every week the model:

1. Downloads fresh market + macro data
2. Builds features
3. Predicts the probability of Gold going UP next week
4. Outputs: BUY / SELL / NEUTRAL bias



