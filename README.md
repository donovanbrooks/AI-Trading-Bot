# Trading Bot Lab

A local research dashboard for evaluating a moving-average trading strategy.
It is deliberately a backtesting and paper-trading tool: it does not send
orders to a broker or handle account credentials.

## Run it locally

Use Python 3.9 or newer. From the project folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open the local dashboard in a browser. Select a ticker,
history range, moving-average windows, capital, and estimated trading costs.

## What the first app version does

- Downloads daily market data from Yahoo Finance.
- Generates moving-average crossover signals.
- Executes a signal on the *next* session's open, preventing same-bar
  look-ahead bias.
- Displays trade markers, an equity curve, returns, drawdown, and paper
  trade activity.

## Day-trading AI research mode

The **Day-trading AI direction model** uses recent five-minute bars and is a
separate research path from the daily model. It trains in expanding windows,
uses same-session targets, delays execution to the next bar, and forces a
simulated exit before the regular US market closes. It is not connected to
automatic order submission.

## Crypto AI research mode

The **Crypto AI direction model** is a separate five-minute, 24/7 model. Use a
Yahoo Finance ticker such as `BTC-USD` for research; use the Alpaca pair
`BTC/USD` only in the manually confirmed paper crypto-buy form. Strategy
signals never submit paper or live orders automatically.

## Trade-quality gate

Every backtest can cap its position size, new entries per day, and daily loss.
These settings are risk controls, not a promise of improved returns. The AI
confidence threshold remains a separate filter for AI strategies.

## Next milestones

1. Move the existing AI model into a time-series, walk-forward evaluation
   pipeline.
2. Store backtest runs and paper trades in SQLite.
3. Add risk-based position sizing, stop-loss logic, and tests.
4. Connect a paper broker only after the research results are validated.

## Disclaimer

This project is for educational and research purposes. Historical backtests
do not predict future results and this is not financial advice.

## Alpaca paper-account connection

The dashboard can check an Alpaca **paper** account and submit a deliberately
limited manual paper buy. It never enables live trading. Install the updated
dependencies, then create a local credential file:

```bash
cp .env.example .env
```

In `.env`, replace the two placeholder values with the API key ID and secret
from the Alpaca paper dashboard. Keep `ALPACA_PAPER=true`. The `.env` file is
ignored by Git and must never be committed.

The manual buy form is limited to a US-equity market order from $1 to $25. It
requires an explicit checkbox confirmation and blocks orders when the regular
US market is closed.

## International research screener

The Top 10 screener can optionally rank international stocks using Twelve
Data's end-of-day market-data API. Create a Twelve Data account, generate an
API key, and add it only to your local `.env` file:

```text
TWELVE_DATA_API_KEY=your_key_here
```

In the dashboard choose **Top 10 US research screener**, then select
**International stocks (Twelve Data)**. Use provider-qualified symbols such
as `SBIN:NSE`, and confirm every listing, exchange, and currency in Twelve
Data before relying on the result. Rankings compare percentage-based trend and
risk measures; they are research shortlists, not investment advice or order
signals.

## Private deployment

The app uses Supabase managed email/password accounts. Add `SUPABASE_URL` and
`SUPABASE_ANON_KEY` to `.env`, then create an account from the app's sign-up
screen. Configure the password-reset redirect URL in Supabase Auth to match
`APP_BASE_URL` before deploying.

For managed cloud storage, run
[`supabase/migrations/001_managed_app_storage.sql`](supabase/migrations/001_managed_app_storage.sql)
once in the Supabase SQL Editor. It creates user-isolated tables protected by
row-level security. Keep `SUPABASE_SERVICE_ROLE_KEY` and `APP_ENCRYPTION_KEY`
server-only; neither belongs in a browser or repository.

To run the private app as a container after configuring `.env`:

```bash
docker compose up --build
```

This deployment remains paper-only. Do not expose it publicly without a real
identity provider, HTTPS, and a managed secret store.

## Render deployment

The repository includes `render.yaml` for a private Docker deployment. Push
the project to a private Git repository, then in Render choose **New → Blueprint**
and select that repository. Add the values marked `sync: false` as Render
environment secrets; do not upload `.env`. After Render provides its HTTPS
address, set `APP_BASE_URL` to that address and add the same address under
Supabase **Authentication → URL Configuration** as the Site URL and a Redirect
URL. Keep `ALPACA_PAPER=true`.
