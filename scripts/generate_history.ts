/**
 * Generates price history from Gomer Trading's own practice-market model, for backtesting.
 *
 * It imports the real MarketModel from the Gomer Trading source (Project 2), steps it at
 * the same half-second tick the app uses, and rebuilds quotes and one-minute bars with the
 * same rules as backend/src/market/simulatedProvider.ts. Nothing here touches the running
 * app or its database.
 *
 *   ../Project\ 2/node_modules/.bin/tsx scripts/generate_history.ts --sessions 120 --seed 7 --out data/practice
 *
 * With --sessions the market keeps US exchange hours, as the app's practice market does by
 * default: 390 one-minute bars a day from 13:30 UTC, weekdays only, and each morning the
 * model's reopen() moves prices overnight (more after a weekend). With --days it never
 * closes, which is the app's 'always' setting.
 *
 * One CSV per symbol. Beyond OHLCV each bar carries what a backtest needs to apply the
 * exchange's fill rules exactly: the lowest/highest bid and ask seen inside the minute, the
 * closing bid and ask, and volume split by which side of the quote the trade printed on.
 * `mood` is the model's hidden regime at the close; it is written for diagnostics only and
 * the strategy never reads it.
 */
import fs from 'node:fs';
import path from 'node:path';
import { MarketModel, seededRandom } from '../../Project 2/backend/src/market/model.js';
import { LISTINGS } from '../../Project 2/backend/src/market/universe.js';

const args = new Map<string, string>();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i]!.replace(/^--/, ''), process.argv[i + 1] ?? '');

const sessions = Number(args.get('sessions') ?? 0);
const days = Number(args.get('days') ?? 90);
const seed = Number(args.get('seed') ?? 7);
const outDir = args.get('out') ?? 'data/practice';
const wanted = (args.get('symbols') ?? 'AAPL,META,NVDA,AMD,TSLA,PLTR,COIN,GME,SHOP,MU,JPM,SPY').split(',');
/** 2026-06-01T00:00:00Z. Only the spacing matters; the date is arbitrary. */
const startSec = Math.floor(Number(args.get('start') ?? 1780272000) / 60) * 60; // bars open on whole minutes, as the app's do

const TICKS_PER_MINUTE = 120;
const round2 = (n: number) => Math.round(n * 100) / 100;

// Only the wanted symbols are simulated. A symbol depends on the market mood, its sector and
// its own state, never on other symbols, so leaving the rest out changes nothing but speed.
const model = new MarketModel(LISTINGS.filter((l) => wanted.includes(l.symbol)), seededRandom(seed));
const micro = seededRandom(seed * 7919 + 13);

interface Forming {
  open: number; high: number; low: number; close: number; volume: number;
  bidLow: number; bidHigh: number; askLow: number; askHigh: number; bidClose: number; askClose: number;
  buyVolume: number; sellVolume: number;
}

fs.mkdirSync(outDir, { recursive: true });
const HEADER = 'time,open,high,low,close,volume,bid_low,bid_high,ask_low,ask_high,bid_close,ask_close,buy_volume,sell_volume,mood\n';
const streams = new Map(wanted.map((s) => [s, fs.createWriteStream(path.join(outDir, `${s}.csv`))]));
for (const stream of streams.values()) stream.write(HEADER);
const forming = new Map<string, Forming | null>(wanted.map((s) => [s, null]));
const lastClose = new Map<string, number>();
const buffers = new Map<string, string[]>(wanted.map((s) => [s, []]));

// Let the model run for an hour first so the opening state is not special.
for (let i = 0; i < 60 * TICKS_PER_MINUTE; i++) model.step();

const SESSION_MINUTES = 390;
const minutes = sessions > 0 ? sessions * SESSION_MINUTES : days * 24 * 60;
/** Session mode: the calendar day (from the start date) each session falls on, skipping weekends. */
let calendarDay = -1;
let barTime = startSec;
for (let minute = 0; minute < minutes; minute++) {
  if (sessions > 0 && minute % SESSION_MINUTES === 0) {
    const previous = calendarDay;
    do calendarDay++;
    while ([0, 6].includes(new Date((startSec + calendarDay * 86400) * 1000).getUTCDay()));
    // The opening bell, as simulatedProvider.openSession() rings it.
    const nights = previous < 0 ? 1 : calendarDay - previous;
    model.reopen(Math.min(1.6, 1 + 0.15 * (nights - 1)));
    for (const symbol of wanted) lastClose.delete(symbol); // the first bar opens at the first trade, not at yesterday's close
    barTime = startSec + calendarDay * 86400 + 13.5 * 3600;
  }
  for (let tick = 0; tick < TICKS_PER_MINUTE; tick++) {
    const { activity } = model.step();
    for (const symbol of wanted) {
      const busy = activity.get(symbol) ?? 1;
      const price = model.priceOf(symbol);
      // requote(): the Level 1 quote around the model price.
      const spread = Math.max(0.01, round2(price * 0.00008 * Math.min(3, Math.max(1, busy))));
      const bid = round2(price - spread / 2);
      const ask = round2(bid + spread);
      const atAsk = micro() >= 0.5;
      const last = atAsk ? ask : bid;
      const lastSize = Math.max(1, Math.round(busy * (20 + micro() * 380)));
      // record(): the one-minute bar.
      const volume = Math.round(lastSize * (1 + busy));
      let bar = forming.get(symbol) ?? null;
      if (!bar) {
        const open = lastClose.get(symbol) ?? last;
        bar = { open, high: Math.max(open, last), low: Math.min(open, last), close: last, volume: 0, bidLow: bid, bidHigh: bid, askLow: ask, askHigh: ask, bidClose: bid, askClose: ask, buyVolume: 0, sellVolume: 0 };
        forming.set(symbol, bar);
      }
      bar.high = Math.max(bar.high, last);
      bar.low = Math.min(bar.low, last);
      bar.close = last;
      bar.volume += volume;
      bar.bidLow = Math.min(bar.bidLow, bid);
      bar.bidHigh = Math.max(bar.bidHigh, bid);
      bar.askLow = Math.min(bar.askLow, ask);
      bar.askHigh = Math.max(bar.askHigh, ask);
      bar.bidClose = bid;
      bar.askClose = ask;
      if (atAsk) bar.buyVolume += volume;
      else bar.sellVolume += volume;
    }
  }
  const time = sessions > 0 ? barTime + (minute % SESSION_MINUTES) * 60 : startSec + minute * 60;
  const mood = model.mood;
  for (const symbol of wanted) {
    const b = forming.get(symbol)!;
    buffers.get(symbol)!.push(
      `${time},${b.open},${b.high},${b.low},${b.close},${b.volume},${b.bidLow},${b.bidHigh},${b.askLow},${b.askHigh},${b.bidClose},${b.askClose},${b.buyVolume},${b.sellVolume},${mood}\n`,
    );
    lastClose.set(symbol, b.close);
    forming.set(symbol, null);
  }
  if (minute % 390 === 389) {
    for (const symbol of wanted) {
      streams.get(symbol)!.write(buffers.get(symbol)!.join(''));
      buffers.set(symbol, []);
    }
  }
}
for (const symbol of wanted) {
  streams.get(symbol)!.write(buffers.get(symbol)!.join(''));
  streams.get(symbol)!.end();
}
console.log(`Wrote ${sessions > 0 ? `${sessions} exchange-hours sessions` : `${days} days`} of one-minute bars for ${wanted.length} symbols to ${outDir} (seed ${seed})`);
