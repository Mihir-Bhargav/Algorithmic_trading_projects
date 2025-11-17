#!/usr/bin/env python3
"""
robust_option_backtest.py

Robust Black-Scholes option backtester tuned for multiple trades/day and
defensive against missing / malformed CSV data.

Assumptions about CSV (common column names):
 - 'quote_date' (date)
 - 'expiration' (date)
 - 'strike' (float)
 - 'type' (call/put) -- we filter to calls in this example
 - 'bid', 'ask' (float) -- option quote prices per share
 - 'implied_volatility' (float) -- e.g. 0.25 for 25% (optional; fallback used)
 - 'delta' (float) optional
 - Spot/underlying price may be present under several names:
    'underlying_price', 'underlying', 'spot', 'close', 'last', 'underlying_last'
"""

import pandas as pd
import numpy as np
import math
import logging
from datetime import datetime, timedelta

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- Constants ----------
CONTRACT_MULTIPLIER = 100  # price quoted per share; 1 contract = 100 shares

class OptionBacktestCSV:
    def __init__(self, csv_path, amount):
        self.csv_path = csv_path
        self.initial_balance = float(amount)
        self.current_balance = float(amount)
        self.positions = {}  # {contract_id: {...}}
        self.trades = 0
        self.errors = []
        self.bad_rows = 0
        self.get_data(csv_path)

    def get_data(self, csv_path):
        # Read CSV robustly and coerce expected columns
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"Failed to read CSV '{csv_path}': {e}")
            raise

        # Parse dates defensively
        for col in ['quote_date', 'expiration']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    logging.warning(f"Column '{col}' exists but could not be parsed fully as datetimes. Coercing errors.")
                    df[col] = pd.to_datetime(df[col], errors='coerce')

        # Filter to calls (case-insensitive), if 'type' exists
        if 'type' in df.columns:
            df = df[df['type'].astype(str).str.lower() == 'call']
        # Ensure essential columns exist, otherwise create with NaNs (will be filtered later)
        for col in ['strike', 'bid', 'ask', 'implied_volatility', 'delta']:
            if col not in df.columns:
                df[col] = np.nan

        # Coerce numeric columns
        num_cols = ['strike', 'bid', 'ask', 'implied_volatility', 'delta']
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Reset index and store
        df = df.sort_values(['quote_date', 'expiration', 'strike'], na_position='last').reset_index(drop=True)
        self.data = df
        logging.info(f"Loaded {len(self.data)} rows from CSV.")

    def normal_cdf(self, x):
        # Use math.erf for numerical stability: Phi(x) = 0.5*(1 + erf(x / sqrt(2)))
        try:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        except Exception as e:
            # In case of a weird input, fallback to numeric safe clamps
            if np.isfinite(x):
                xv = float(x)
                return 0.5 * (1.0 + math.erf(xv / math.sqrt(2.0)))
            else:
                return 0.0

    def black_scholes_call(self, S, K, T, r, sigma):
        # returns (call_price_per_share, delta)
        try:
            if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
                return (np.nan, np.nan)
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            N_d1 = self.normal_cdf(d1)
            N_d2 = self.normal_cdf(d2)
            call = S * N_d1 - K * math.exp(-r * T) * N_d2
            return (call, N_d1)
        except Exception as e:
            return (np.nan, np.nan)

    def _get_spot(self, row):
        # Look for common underlying/spot column names
        for col in ['underlying_price', 'underlying', 'spot', 'close', 'last', 'underlying_last']:
            if col in row and not pd.isna(row[col]) and float(row[col]) > 0:
                return float(row[col])
        return None

    def run_strategy(self,
                     r=0.01,
                     daily_trade_target=7,
                     per_trade_allocation=0.02,
                     min_T_days=7,
                     max_T_days=45,
                     min_iv=0.02,
                     max_bidask_spread_pct=0.30,
                     edge_pct_threshold=0.02,
                     delta_min=0.25,
                     delta_max=0.70,
                     tp=0.40,
                     sl=0.25,
                     close_before_expiry_days=2,
                     max_contracts_per_trade=100):
        """
        Runs the backtest with robust error handling and conservative defaults.
        Tweak parameters at call-time.
        """
        logging.info("Starting strategy run")
        if self.data.empty:
            logging.error("No data available to run strategy.")
            return

        # Ensure quote_date present and sorted
        if 'quote_date' not in self.data.columns:
            logging.error("CSV missing 'quote_date' column - cannot proceed.")
            return

        grouped = self.data.groupby('quote_date')
        # Pre-build price_map for final forced close optimization
        # We'll keep a dict of last-known per-contract price for each contract_id
        last_known_price = {}

        for date, day_df in grouped:
            date = pd.Timestamp(date)
            # build day price map (contract_id -> mid per-share)
            day_price_map = {}
            for _, row in day_df.iterrows():
                try:
                    cid = f"{pd.Timestamp(row['expiration']).date()}_{row['strike']}"
                except Exception:
                    continue
                bid = row.get('bid', np.nan)
                ask = row.get('ask', np.nan)
                # compute mid if possible (per-share)
                price_per_share = None
                if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
                    price_per_share = (bid + ask) / 2.0
                else:
                    # take whichever is >0
                    price_per_share = max(bid if not pd.isna(bid) else 0.0, ask if not pd.isna(ask) else 0.0)
                day_price_map[cid] = price_per_share
                # update global last known per-contract
                if price_per_share and price_per_share > 0:
                    last_known_price[cid] = price_per_share * CONTRACT_MULTIPLIER

            # 1) Update existing positions for TP/SL/expiry closings using today's prices
            for cid in list(self.positions.keys()):
                pos = self.positions[cid]
                # get today's price if available
                mkt_per_contract = day_price_map.get(cid, None)
                if mkt_per_contract is None or mkt_per_contract == 0:
                    # no quote today for this contract: skip TP/SL checks
                    continue
                mkt = mkt_per_contract * CONTRACT_MULTIPLIER
                entry = pos['entry_price']
                try:
                    # TP
                    if mkt >= entry * (1.0 + pos['tp']):
                        self.current_balance += mkt * pos['contracts']
                        logging.info(f"{date.date()} | TAKE-PROFIT SELL {cid} at {mkt:.2f} (entry {entry:.2f})")
                        self.trades += 1
                        del self.positions[cid]
                    # SL
                    elif mkt <= entry * (1.0 - pos['sl']):
                        self.current_balance += mkt * pos['contracts']
                        logging.info(f"{date.date()} | STOP-LOSS SELL {cid} at {mkt:.2f} (entry {entry:.2f})")
                        self.trades += 1
                        del self.positions[cid]
                    else:
                        # forced close near expiry
                        if (pos['expiration'] - date).days <= close_before_expiry_days:
                            self.current_balance += mkt * pos['contracts']
                            logging.info(f"{date.date()} | CLOSE-NEAR-EXPIRY {cid} at {mkt:.2f}")
                            self.trades += 1
                            del self.positions[cid]
                except Exception as e:
                    # log and keep the position (do not crash)
                    self.errors.append(f"Error while evaluating existing position {cid} on {date}: {e}")

            # 2) Build candidates for this day
            candidates = []
            for idx, row in day_df.iterrows():
                try:
                    # defensive extraction & validation
                    if pd.isna(row.get('expiration')) or pd.isna(row.get('quote_date')) or pd.isna(row.get('strike')):
                        self.bad_rows += 1
                        continue

                    S = self._get_spot(row)
                    if S is None:
                        # skip rows without spot - safer than using strike fallback
                        self.bad_rows += 1
                        continue

                    K = float(row['strike'])
                    T_days = int((pd.Timestamp(row['expiration']) - pd.Timestamp(row['quote_date'])).days)
                    # skip negative / zero T
                    if T_days <= 0:
                        continue
                    T = T_days / 365.0
                    sigma = float(row['implied_volatility']) if not pd.isna(row.get('implied_volatility')) else 0.25
                    if sigma < min_iv:
                        continue

                    # compute per-share mid and spread tolerance
                    bid = row.get('bid', np.nan)
                    ask = row.get('ask', np.nan)
                    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
                        market_price_per_share = (bid + ask) / 2.0
                        bidask_spread_pct = (ask - bid) / market_price_per_share if market_price_per_share>0 else 1.0
                    else:
                        market_price_per_share = max(bid if not pd.isna(bid) else 0.0, ask if not pd.isna(ask) else 0.0)
                        # if only a single side exists, treat spread as 0 (conservative)
                        bidask_spread_pct = 0.0

                    if market_price_per_share <= 0:
                        continue
                    if bidask_spread_pct > max_bidask_spread_pct:
                        continue
                    # theoretical BS (per-share)
                    bs_per_share, bs_delta = self.black_scholes_call(S, K, T, r, sigma)
                    if np.isnan(bs_per_share) or np.isnan(bs_delta):
                        continue
                    # convert to per-contract
                    market_price = market_price_per_share * CONTRACT_MULTIPLIER
                    bs_price = bs_per_share * CONTRACT_MULTIPLIER
                    edge = bs_price - market_price
                    if market_price == 0:
                        continue
                    edge_pct = edge / market_price
                    contract_id = f"{pd.Timestamp(row['expiration']).date()}_{row['strike']}"
                    csv_delta = row.get('delta', np.nan)
                    delta_use = float(csv_delta) if not pd.isna(csv_delta) else float(bs_delta)
                    # time-to-expiry filter
                    if T_days < min_T_days or T_days > max_T_days:
                        continue

                    candidates.append({
                        'edge': edge,
                        'edge_pct': edge_pct,
                        'contract_id': contract_id,
                        'row': row,
                        'bs_price': bs_price,
                        'delta': delta_use,
                        'market_price': market_price,
                        'expiration': pd.Timestamp(row['expiration']),
                    })
                except Exception as e:
                    # catch row-level exceptions to avoid intermittent crashes
                    self.errors.append(f"Row processing error on {row.get('quote_date')}: {e}")
                    self.bad_rows += 1
                    continue

            # sort and pick top candidates
            candidates = sorted(candidates, key=lambda x: x['edge_pct'], reverse=True)
            entered_today = 0
            for c in candidates:
                if entered_today >= daily_trade_target:
                    break
                cid = c['contract_id']
                mp = c['market_price']  # per-contract
                if cid in self.positions:
                    continue
                # apply entry rules
                if c['edge_pct'] >= edge_pct_threshold and delta_min <= c['delta'] <= delta_max:
                    # position sizing: per_trade_allocation * current_balance
                    allocation = max(0.0001, per_trade_allocation) * self.current_balance
                    contracts = int(max(1, allocation // mp))
                    # absolute cap
                    if contracts > max_contracts_per_trade:
                        contracts = max_contracts_per_trade
                    cost = mp * contracts
                    # safety: don't spend more than 10% of current balance
                    if cost > 0.10 * self.current_balance:
                        contracts = int((0.10 * self.current_balance) // mp)
                        if contracts < 1:
                            continue
                        cost = mp * contracts
                    if contracts < 1:
                        continue
                    # execute buy
                    self.current_balance -= cost
                    self.positions[cid] = {
                        'entry_price': mp,
                        'contracts': contracts,
                        'entry_date': date,
                        'tp': tp,
                        'sl': sl,
                        'strike': c['row']['strike'],
                        'expiration': c['expiration']
                    }
                    self.trades += 1
                    entered_today += 1
                    logging.info(f"{date.date()} | BUY {cid} x{contracts} at {mp:.2f} | Delta: {c['delta']:.2f} | BS: {c['bs_price']:.2f} | Edge%: {c['edge_pct']*100:.2f}")

        # After processing all dates, forced-close remaining positions using last known per-contract prices
        if self.positions:
            for cid in list(self.positions.keys()):
                last_price = last_known_price.get(cid, None)
                if last_price and last_price > 0:
                    pos = self.positions[cid]
                    self.current_balance += last_price * pos['contracts']
                    logging.info(f"FINAL FORCED SELL {cid} at {last_price:.2f}")
                    self.trades += 1
                    del self.positions[cid]
                else:
                    # no price data for this contract — keep as unrealized (more conservative than forcing to 0)
                    logging.warning(f"No final price for {cid}; leaving unrealized (no forced close).")

        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100.0
        logging.info(f"Net performance (%) = {perf:.2f}")
        logging.info(f"Number of trades executed = {self.trades}")
        logging.info(f"Ending balance = {self.current_balance:.2f}")
        logging.info(f"Rows skipped / bad = {self.bad_rows}, runtime errors recorded = {len(self.errors)}")
        if self.errors:
            logging.debug("Sample errors:")
            for e in self.errors[:20]:
                logging.debug("  " + str(e))

# --------- Example usage ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Robust Black-Scholes Option Backtest")
    parser.add_argument("2013-06-20options.csv", type=str, required=True, help="Path to options CSV")
    parser.add_argument("10000", type=float, default=10000, help="Starting cash amount")
    args = parser.parse_args()

    bt = OptionBacktestCSV(csv_path=args.csv, amount=args.amount)
    bt.run_strategy(
        r=0.01,
        daily_trade_target=7,
        per_trade_allocation=0.02,
        min_T_days=7,
        max_T_days=45,
        min_iv=0.02,
        max_bidask_spread_pct=0.30,
        edge_pct_threshold=0.02,
        delta_min=0.25,
        delta_max=0.70,
        tp=0.40,
        sl=0.25,
        close_before_expiry_days=2,
        max_contracts_per_trade=100
    )



