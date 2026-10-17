# AT — algorithmic trader for Gomer Trading

A systematic strategy, a backtester with walk-forward validation, and a live bot that trades
through Gomer Trading's REST and WebSocket API under the name **AT**.

No real money is involved anywhere: the bot trades the paper accounts of the Gomer Trading
app, and the research runs on that app's own practice-market model.

## What the research found

The brief was a regime-filtered **mean-reversion** strategy: fade stretched prices
(RSI(2) extreme, more than 2.5 deviations from VWAP, absorption at support/resistance) when
the regime is ranging, take profit at the mean, stop at 3.5 deviations.

That strategy is built exactly as written, and on the practice market **it loses**. The
practice market trends: one-minute returns are positively correlated with the next ones, a
stretched price is far more likely to keep going than to come back, and the Hurst exponent
sits near 0.6 almost all the time. The regime filter does its job (it mostly keeps the
strategy out), and the trades that get through lose.

So the strategy has two playbooks that share every measurement, and the regime gate decides
which side of a stretch to take:

| playbook | what it does | on the practice market |
|---|---|---|
| `reversion` | the brief: fade a stretch in a ranging regime | loses; no candidate was profitable |
| `continuation` | trade **with** a stretch in a trending regime, quick target, hard stop | wins about 85% of trades |
| `adaptive` | let the regime gate pick between the two, bar by bar | profitable, but worse than pure continuation |

The walk-forward optimiser chose `continuation` in every fold. The numbers, the overfitting
checks and the things that would make them wrong are in
[reports/backtest_report.md](reports/backtest_report.md).

### Second round: after the first live trades

AT's first live trades (two short trials, 22 trades) won 63.6%, well under the backtest.
Replaying the same minutes, the backtester made the same trades and lost the same ones, so
the execution was right and the rules were what needed work. Studying 31,000 backtest trades
showed where the losers gather: where the directional index barely leans the trade's way,
where the signal bar was pushed back, and on volume climaxes. Filters for those were added
and chosen by a second walk-forward. On a holdout the settings had never seen:

| | before | after |
|---|---|---|
| success rate | 85.9% | **91.4%** |
| a bad half-hour (1 in 20 is worse) | 62% | 76% |
| half-hours under 70% | 12.1% | 1.0% |
| worst session of 60 | 77% | 76% |
| average trade | +12.9 bps | +16.3 bps |
| trades per session | about 278 | about 94 |

Of the 12 trial trades that could be re-examined, 6 won and 6 lost; the new filters would
have skipped all 12, because that stretch never had a strong enough trend. The cost of the
higher success rate is fewer trades, and with them less total profit in the backtest.

Between the two rounds the app's practice market changed: it now keeps US exchange hours
and opens each day with an overnight price jump. Indicators are computed with those jumps
taken out, so the first bar of a day is not read as a huge one-minute move, and nothing is
opened in a session's last three minutes or held over a close. `configs/at_v1.json` is the
first round's configuration, kept for comparison.

**Read these before trusting the numbers:**

- They describe the *practice market*, whose trends are much stronger than a real market's.
  On real prices (a market-data key in the app) expect little or no edge from either playbook.
  The live bot re-measures the regime on every bar, but its settings were tuned on the
  practice market and would need a fresh research run on real history.
- A high success rate is bought with a poor payoff ratio: the average loss is larger than
  the average win. That is the nature of a small target and a wider stop, not a free lunch.
  The report shows that random entries with the same exits still win about 60% of the time
  and lose money; the signal is what turns that into a profit.
- The backtest ignores AT's own effect on prices. In the practice market orders move the
  price, so results at size will differ.
- No success rate can be promised live. The backtest says about nine in ten; the first
  sessions will show how much of that carries over.
- Returns are quoted on a fixed $100,000 with no compounding. The absolute figures are large
  because the practice market is that easy to trend-follow; they say more about the simulator
  than about the strategy.

## Layout

```
mrbot/
  config.py         every parameter (StrategyConfig) and the exchange's costs (CostModel)
  indicators.py     RSI, ATR, ADX, Bollinger width, rolling/session VWAP bands, Hurst, volume delta
  strategy.py       RegimeMeanReversion: regime gate, entry confluence, entry orders, exits
  risk.py           fixed-fractional / fractional-Kelly sizing, daily-loss circuit breaker
  backtest.py       event-driven backtester: the exchange's fill rules, slippage, fees
  walkforward.py    walk-forward optimisation and the overfitting checks
  metrics.py        statistics, shared by backtests and live sessions
  review.py         after a session: why each losing trade lost, and whether the rules should change
  gateway.py        GatewayClient: the app's bot gateway (API key, JSON over WebSocket)
  atbot.py          GatewayBot: AT as it runs in the app, through the gateway
  exchange.py       GomerClient: the app's trader API (login, bracket orders)
  bot.py            TradingBot for the trader API, and the session Journal both bots use
  api.py            local FastAPI control panel for a running bot
  mock_exchange.py  a stand-in Gomer Trading server for end-to-end tests
scripts/
  generate_history.ts   price history from Gomer Trading's real market model
  run_research.py       walk-forward, holdout and every check → reports/ and configs/at.json
  admit_at.py           admit AT to the app as a trading program and save its API key
  run_at.py             run AT for a session through the gateway and write its report
  run_bot.py            the same for the trader API (needs a trader account, not used by AT)
configs/at.json         the configuration the research chose (what the bot trades)
configs/at_v1.json      the first round's configuration, kept for comparison
reports/                backtest_report.md, backtest_results.json, equity.png
runs/                   one folder per live session: report.md, trades.csv, equity.csv, stats.json
tests/                  indicators (no look-ahead), fill rules, both bots
```

## How the pieces fit

**Strategy.** `RegimeMeanReversion` is the single source of trading rules. The backtester and
the bot both call the same four methods (`indicators`, `signals`, `entry`, `manage`), and a
test checks that the bot's trailing window gives the same signals as a full-history backtest.

1. *Regime gate* on a higher timeframe: Bollinger width low in its own range (volatility
   consolidating) and ADX below a threshold or Hurst below 0.5 (ranging). The complement of
   the same measures is "trending".
2. *Entry confluence*: RSI(2) at an extreme **and** price beyond the VWAP deviation band
   **and** order flow (heavy volume that was absorbed → fade; heavy volume that carried the
   bar → follow).
3. *Execution*: limit orders only. Passive entries join the touch and earn the spread;
   marketable ones cross it with a capped limit. Size is fixed-fractional risk, replaced by
   a quarter-Kelly stake once 40 trades have closed, always between a floor and a cap.
4. *Risk*: take profit at the mean (reversion) or a fraction of ATR (continuation); a hard
   stop sent to the exchange with the entry; a time stop; and a circuit breaker that
   flattens and halts after a 2% loss in a session.

**Backtester.** One bar at a time across all symbols, sharing equity and the breaker.
Orders decided at a bar's close cannot fill before the next bar. A resting limit fills only
when the *opposite* quote reaches it (Gomer Trading's rule). If a bar could have hit both
stop and target, the stop is assumed first. Takers pay slippage; stops pay extra.

**Walk-forward.** A few hundred candidates drawn from a fixed grid. Choose on four weeks,
test on the week after, slide forward, stitch the tests together. Then the parameters are
frozen and run once on a separate history from a different random seed. Reported alongside:
walk-forward efficiency, in-sample/out-of-sample rank correlation, probability of overfit,
how steady the success rate is across half-hours and sessions, a random-entry benchmark, an
ablation of each filter, and a random-walk market on which the strategy must not make money.

**Bot.** AT is connected to the app the way the app means trading programs to connect: it is
admitted in the Admin console's *Trading programs (bots)* list under the label **AT**, has
its own account (`BOT-…`) and API key, and trades in a competition of its own, **AT
Sandbox (ongoing)**, so its results do not move anyone else's leaderboard. `GatewayBot` logs on to the
bot gateway, streams best bid and offer, and on each finished minute takes in the new bar
and decides.

- An entry is a limit order; a marketable one is sent immediate-or-cancel.
- The moment any of it fills, a STOP order for exactly those shares rests at the exchange.
  It is there whether or not AT is still running.
- AT watches the quote for the target and sends an immediate-or-cancel limit at the target
  price. (The gateway has no bracket orders, and a take-profit resting beside a stop could
  sell twice.)
- Once a minute AT compares its picture with the exchange's account. A position it does not
  recognise is closed and journalled; an order it does not recognise is cancelled.
- The gateway has no price history, so bars come from the app's chart API, read with the
  same local sign-in the desktop app uses for its owner. That is why AT has to run on the
  Mac the app runs on.

The `Journal` records every trade and writes a report that leads with the success rate and
sets the session beside the backtest's expectation. `TradingBot` is an earlier integration
through the trader API (login, server-side bracket orders); it is kept and tested, but AT
does not use it.

## Running it

Python 3.12+ and the packages in `requirements.txt` (a virtual environment is already in
`.venv`).

```bash
.venv/bin/python -m pytest tests -q
```

Regenerate the price history (needs the Gomer Trading source in `../Project 2`; a few
minutes each, about 80 MB together):

```bash
"../Project 2/node_modules/.bin/tsx" scripts/generate_history.ts --sessions 120 --seed 7 --out data/sessions
```

```bash
"../Project 2/node_modules/.bin/tsx" scripts/generate_history.ts --sessions 60 --seed 23 --out data/sessions_holdout --start 1791158400
```

Re-run the research (about four minutes; rewrites `reports/` and `configs/at.json`):

```bash
.venv/bin/python scripts/run_research.py
```

Admit AT to the app on this Mac (the app must be open). Safe to run again; it prints what
it did and never prints the key:

```bash
.venv/bin/python scripts/admit_at.py
```

Trade for half an hour, starting now, with a local control panel:

```bash
.venv/bin/python scripts/run_at.py --minutes 30 --api-port 8787
```

or wait for the US market open and trade the whole session:

```bash
.venv/bin/python scripts/run_at.py --us-session
```

Either way AT closes everything at the end and writes `runs/<date>/report.md`, `trades.csv`,
`equity.csv` and `stats.json`; `runs/latest.md` is always the newest report. Ctrl-C stops it
cleanly (it cancels, flattens and still writes the report). While it runs,
`http://127.0.0.1:8787/stats` shows the session's statistics so far.

To have AT on another competition's leaderboard instead of its sandbox, change its sandbox
in the Admin console (*Trading programs*), or:

```bash
.venv/bin/python scripts/admit_at.py --competition "Demo Trading Challenge"
```

## Automatic sessions

AT starts by itself. A launch agent, `~/Library/LaunchAgents/com.gomertrading.at.plist`
(installed 14 October 2026 at the user's request), starts `run_at.py --us-session --once`
at 6:50 pm IST every weekday and whenever you log in. It waits for the US market to open
(7:00 pm IST while the US is on summer time, 8:00 pm in winter), trades until three minutes
before the close, closes everything, writes the report and loss review, shows a notification
with the success rate, and exits until the next start. Market holidays are skipped by the
app's own calendar. If it crashes, launchd restarts it and it picks up the session where it
left off.

It still needs the Mac on and awake and the Gomer Trading app open during the session; if
the app is opened late, AT starts when it finds it. Its log is `runs/at-agent.log`.

Only one AT runs at a time. **Start AT.command** still works for a session by hand, and does
nothing if AT is already running.

Turn the automatic start off:

```bash
launchctl bootout gui/$(id -u)/com.gomertrading.at
```

and back on:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gomertrading.at.plist
```

AT trades in **AT Sandbox (ongoing)**, which runs until 31 December 2099. (The app cannot
change a competition's end date, so on 14 October 2026 this replaced the first sandbox, "AT
Sandbox", which ends on 9 November and keeps the two trial runs. AT started the new one with
a fresh $100,000.)

### AT on the Demo Trading Challenge too

Since 15 October 2026 AT also trades in the **Demo Trading Challenge**, against the ten
practice traders, so it shows on that competition's leaderboard as "AT". It is the same AT
account (`BOT-B720F4`) with a second key, listed in the Admin console as *AT (Demo Trading
Challenge)*; the key is `AT_API_KEY_DEMO` in `.env`. The same process trades both
competitions with the same rules, each from its own $100,000. The Demo results go to
`runs/<date>-demo/` and `runs/latest-demo.md`, with AT's place on the leaderboard at the end.

Two things to know: AT now sends every order twice into the same practice market, so the
second copy's fills are slightly worse than the first's; and the leaderboard lists the top
ten of eleven, so AT appears on it once it is ahead of at least one trader. The Demo Trading
Challenge ends on 8 November 2026; after that the app refuses AT's orders there and AT
stops opening trades in it. Removing the *AT (Demo Trading Challenge)* program in the Admin
console (or deleting `AT_API_KEY_DEMO` from `.env`) stops the second copy.

## After every session: the loss review

When a session ends, AT replays the session's own minutes through the backtester and sorts
every losing trade into one of three kinds (`mrbot/review.py`):

- **expected**: the backtester takes the same trade and loses it too. The rules did what
  they were built to do.
- **execution**: the backtester wins a trade AT lost live. Something between the rules and
  the exchange went differently, and that is a fault to find.
- **unplanned**: the backtester would not have taken the trade at all.

It then asks whether the session's success rate fell short of the long-run figure by more
than luck allows. The write-up is appended to the session's `report.md`, the detail goes to
`loss_review.json`, and one line per session is added to `runs/reviews.jsonl`.

The rules are changed when that record says so: execution faults, or a shortfall that
chance does not explain. They are not changed after each losing trade. A strategy that wins
nine times in ten still loses one in ten by design, and re-fitting it to every loss would
tune it to noise and make it worse. When the record does call for a change, re-run the
research; the walk-forward and the holdout decide what the new settings are.

## Limits

- **AT must run on the Mac the app runs on**, because it reads price history through the
  desktop app's local sign-in. A bars call on the gateway itself would remove that.
- **The target is not a resting order.** AT has to be running and connected to take profit;
  if it is not, positions sit behind their stop orders until it is back.
- **The stand-in exchange speaks the trader API, not the gateway.** The gateway bot is
  covered by scripted tests of its order handling and by live sessions in the sandbox.
- **AT needs about two hours of one-minute history** (128 bars). The app keeps bars only for
  the hours it was open, so after a quiet spell AT watches until it has enough, and says so
  in the session report.
- **Results are the practice market's.** See the top of this file.
