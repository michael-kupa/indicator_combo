import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ─────────────────────────────────────────
# Indicator calculation functions
# ─────────────────────────────────────────

def moving_average(data, window):
    return data.rolling(window=window).mean()

def ichimoku_cloud(data):
    nine_high  = data['High'].rolling(9).max()
    nine_low   = data['Low'].rolling(9).min()
    conversion = (nine_high + nine_low) / 2
    t26h = data['High'].rolling(26).max()
    t26l = data['Low'].rolling(26).min()
    base = (t26h + t26l) / 2
    span_a  = ((conversion + base) / 2).shift(26)
    f52h    = data['High'].rolling(52).max()
    f52l    = data['Low'].rolling(52).min()
    span_b  = ((f52h + f52l) / 2).shift(26)
    lagging = data['Close'].shift(-26)
    return conversion, base, span_a, span_b, lagging

def envelopes(data, window=20, deviation=0.02):
    ma = data.rolling(window).mean()
    return ma * (1 + deviation), ma * (1 - deviation)

def exponential_moving_average(data, window):
    return data.ewm(span=window, adjust=False).mean()

def parabolic_sar(data, iaf=0.02, maxaf=0.2):
    length = len(data)
    high   = list(data['High'])
    low    = list(data['Low'])
    close  = list(data['Close'])
    psar   = close[:]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True; af = iaf; hp = high[0]; lp = low[0]
    for i in range(2, length):
        psar[i] = psar[i-1] + af * ((hp if bull else lp) - psar[i-1])
        reverse = False
        if bull:
            if low[i] < psar[i]:
                bull, reverse, psar[i], lp, af = False, True, hp, low[i], iaf
        else:
            if high[i] > psar[i]:
                bull, reverse, psar[i], hp, af = True, True, lp, high[i], iaf
        if not reverse:
            if bull:
                if high[i] > hp: hp = high[i]; af = min(af + iaf, maxaf)
                if low[i-1]  < psar[i]: psar[i] = low[i-1]
                if low[i-2]  < psar[i]: psar[i] = low[i-2]
            else:
                if low[i] < lp: lp = low[i]; af = min(af + iaf, maxaf)
                if high[i-1] > psar[i]: psar[i] = high[i-1]
                if high[i-2] > psar[i]: psar[i] = high[i-2]
        if bull: psarbull[i] = psar[i]
        else:    psarbear[i] = psar[i]
    return psarbull, psarbear

def compute_rsi(data, window):
    delta = data.diff(1)
    gain  = delta.where(delta > 0, 0)
    loss  = -delta.where(delta < 0, 0)
    rs    = gain.rolling(window).mean() / loss.rolling(window).mean()
    return 100 - (100 / (1 + rs))

def macd(data, fast=12, slow=26, signal=9):
    ml = data.ewm(span=fast, adjust=False).mean() - data.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=signal, adjust=False).mean()
    return ml, sl

def on_balance_volume(data):
    return (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

def stochastic_oscillator(data, k=14, d=3):
    low_min  = data['Low'].rolling(k).min()
    high_max = data['High'].rolling(k).max()
    k_line   = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    return k_line, k_line.rolling(d).mean()

def compute_cci(data, window=14):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    ma = tp.rolling(window).mean()
    md = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    return (tp - ma) / (0.015 * md)

def compute_dmi(data, window=14):
    dm_plus  = data['High'].diff().clip(lower=0)
    dm_minus = -data['Low'].diff().clip(upper=0)
    tr = (data['High'] - data['Low']).combine(
         (data['High'] - data['Close'].shift()).abs(), max).combine(
         (data['Low']  - data['Close'].shift()).abs(), max)
    atr      = tr.rolling(window).mean()
    di_plus  = 100 * dm_plus.rolling(window).mean()  / atr
    di_minus = 100 * dm_minus.rolling(window).mean() / atr
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    return di_plus, di_minus, dx.rolling(window).mean()

def compute_wr(data, window=14):
    hmax = data['High'].rolling(window).max()
    lmin = data['Low'].rolling(window).min()
    return -100 * ((hmax - data['Close']) / (hmax - lmin))

def rate_of_change(data, window=25):
    return (data / data.shift(window) - 1) * 100

def compute_momentum(data, window=14):
    return data.diff(window)

def compute_atr(data, window=14):
    tr = (data['High'] - data['Low']).combine(
         (data['High'] - data['Close'].shift()).abs(), max).combine(
         (data['Low']  - data['Close'].shift()).abs(), max)
    return tr.rolling(window).mean()


# ─────────────────────────────────────────
# Signal definitions
# ─────────────────────────────────────────

def get_signals(data):
    return {
        'MA_20 (Trend Following)':           data['Close'] > data['MA_20'],
        'Conversion_Line (Trend Following)': data['Close'] > data['Conversion_Line'],
        'Base_Line (Trend Following)':       data['Close'] > data['Base_Line'],
        'Upper_Env (Trend Following)':       data['Close'] > data['Upper_Env'],
        'EMA_26 (Trend Following)':          data['Close'] > data['EMA_26'],
        'PSAR_Bull (Trend Following)':       data['Close'] > data['PSAR_Bull'],
        'RSI_14 (Mean Reversion)':           data['RSI_14'] > 70,
        'RSI_3 (Mean Reversion)':            data['RSI_3']  > 80,
        'MACD_Line (Trend Following)':       data['MACD_Line'] > data['Signal_Line'],
        'OBV (Volume)':                      data['OBV'] > data['OBV'].shift(),
        'Stoch_K (Mean Reversion)':          data['Stoch_K'] > data['Stoch_D'],
        'CCI_14 (Mean Reversion)':           data['CCI_14'] > 100,
        'DI_Plus (Trend Following)':         data['DI_Plus'] > data['DI_Minus'],
        'WR_14 (Mean Reversion)':            data['WR_14'] < -20,
        'ROC_25 (Momentum)':                 data['ROC_25'] > 0,
        'Momentum_14 (Momentum)':            data['Momentum_14'] > 0,
        'ATR_14 (Volatility)':               data['ATR_14'] > data['ATR_14'].rolling(14).mean(),
    }


# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────

INITIAL_CAPITAL  = 10_000.0
ANNUAL_BOND_RATE = 0.045          # ~4.5% US Treasury approximation
DAILY_BOND_RATE  = ANNUAL_BOND_RATE / 252


# ─────────────────────────────────────────
# Count actual executed trades (no overlap)
# ─────────────────────────────────────────

def count_actual_trades(sig_array: np.ndarray, holding_days: int):
    """
    Given a boolean signal array, return:
      - signals_total: total days signal was True
      - trades_actual: how many trades actually executed (no overlap)
      - skipped: signals that were blocked because we were in a trade
    Also returns trade_mask: array of ints:
      0 = no signal, no trade
      1 = in active ticker trade
      2 = signal fired, trade entered (entry day, earns SPY still)
      -1 = signal fired but BLOCKED (already in trade)
    """
    n              = len(sig_array)
    trade_mask     = np.zeros(n, dtype=np.int8)
    signals_total  = int(sig_array.sum())
    trades_actual  = 0
    skipped        = 0
    in_trade_until = -1

    for i in range(n):
        if sig_array[i]:
            if in_trade_until < i:
                # Execute trade
                trades_actual  += 1
                trade_mask[i]   = 2          # entry day
                in_trade_until  = i + holding_days
            else:
                skipped        += 1
                trade_mask[i]   = -1         # blocked
        elif in_trade_until >= 0 and i <= in_trade_until:
            trade_mask[i] = 1                # inside trade

    return signals_total, trades_actual, skipped, trade_mask


# ─────────────────────────────────────────
# Build equity curve (2 strategies + buyhold + spy + bond)
# ─────────────────────────────────────────

def build_equity_curve(
    signal:               pd.Series,
    ticker_daily_returns: pd.Series,
    spy_daily_returns:    pd.Series,
    holding_days:         int,
    dates:                pd.DatetimeIndex,
    ticker:               str,
) -> tuple[list, list]:
    """
    Returns (equity_rows, daily_table_rows).

    equity_rows: [{date, strat_spy, strat_bond, buyhold, spy, bond,
                   position_spy, position_bond}]
      position: 'ticker' | 'spy' | 'bond' | 'entry' | 'blocked'

    daily_table_rows: same but formatted for the detailed table section
    """
    n   = len(dates)
    tr  = ticker_daily_returns.reindex(dates).fillna(0).values
    sr  = spy_daily_returns.reindex(dates).fillna(0).values
    sig = signal.reindex(dates).fillna(False).values

    bond_daily = np.full(n, DAILY_BOND_RATE)

    # Build both strategies simultaneously
    ret_spy  = np.zeros(n)   # Strategy A: idle → SPY
    ret_bond = np.zeros(n)   # Strategy B: idle → US Bond
    pos_spy  = [''] * n      # position label per day
    pos_bond = [''] * n

    in_trade_until_spy  = -1
    in_trade_until_bond = -1

    for i in range(n):
        # Strategy A (idle=SPY)
        if in_trade_until_spy >= 0 and i <= in_trade_until_spy:
            ret_spy[i]  = tr[i]
            pos_spy[i]  = 'ticker'
        elif sig[i] and in_trade_until_spy < i:
            ret_spy[i]           = sr[i]    # entry day: still SPY
            in_trade_until_spy   = i + holding_days
            pos_spy[i]           = 'entry'
        elif sig[i]:
            ret_spy[i]  = sr[i]
            pos_spy[i]  = 'blocked'
        else:
            ret_spy[i]  = sr[i]
            pos_spy[i]  = 'spy'

        # Strategy B (idle=Bond)
        if in_trade_until_bond >= 0 and i <= in_trade_until_bond:
            ret_bond[i]  = tr[i]
            pos_bond[i]  = 'ticker'
        elif sig[i] and in_trade_until_bond < i:
            ret_bond[i]           = bond_daily[i]   # entry day: still bond
            in_trade_until_bond   = i + holding_days
            pos_bond[i]           = 'entry'
        elif sig[i]:
            ret_bond[i]  = bond_daily[i]
            pos_bond[i]  = 'blocked'
        else:
            ret_bond[i]  = bond_daily[i]
            pos_bond[i]  = 'bond'

    strat_spy_val  = INITIAL_CAPITAL * np.cumprod(1 + ret_spy)
    strat_bond_val = INITIAL_CAPITAL * np.cumprod(1 + ret_bond)
    buyhold_val    = INITIAL_CAPITAL * np.cumprod(1 + tr)
    spy_val        = INITIAL_CAPITAL * np.cumprod(1 + sr)
    bond_val       = INITIAL_CAPITAL * np.cumprod(1 + bond_daily)

    rows = []
    for i in range(n):
        rows.append({
            "date":       dates[i].strftime("%Y-%m-%d"),
            "strat_spy":  round(float(strat_spy_val[i]),  2),
            "strat_bond": round(float(strat_bond_val[i]), 2),
            "buyhold":    round(float(buyhold_val[i]),    2),
            "spy":        round(float(spy_val[i]),        2),
            "bond":       round(float(bond_val[i]),       2),
            # daily returns for coloring
            "ticker_ret": round(float(tr[i]) * 100, 3),
            "spy_ret":    round(float(sr[i]) * 100, 3),
            # position labels
            "pos_spy":    pos_spy[i],
            "pos_bond":   pos_bond[i],
        })
    return rows


# ─────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────

def run_analysis(ticker: str, holding_days: int, years: int) -> dict:
    end   = datetime.today()
    start = end - timedelta(days=365 * years + 100)

    raw = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                      end=end.strftime('%Y-%m-%d'), auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    spy_raw = yf.download("SPY", start=start.strftime('%Y-%m-%d'),
                          end=end.strftime('%Y-%m-%d'), auto_adjust=True)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)

    data = raw.copy()

    # Indicators
    data['MA_20']  = moving_average(data['Close'], 20)
    conv, base, spa, spb, lag = ichimoku_cloud(data)
    data['Conversion_Line'] = conv
    data['Base_Line']       = base
    data['Upper_Env'], data['Lower_Env'] = envelopes(data['Close'])
    data['EMA_26']  = exponential_moving_average(data['Close'], 26)
    psarbull, _     = parabolic_sar(data)
    data['PSAR_Bull']       = psarbull
    data['RSI_14']  = compute_rsi(data['Close'], 14)
    data['RSI_3']   = compute_rsi(data['Close'], 3)
    data['MACD_Line'], data['Signal_Line'] = macd(data['Close'])
    data['OBV']     = on_balance_volume(data)
    data['Stoch_K'], data['Stoch_D'] = stochastic_oscillator(data)
    data['CCI_14']  = compute_cci(data)
    data['DI_Plus'], data['DI_Minus'], data['ADX'] = compute_dmi(data)
    data['WR_14']   = compute_wr(data)
    data['ROC_25']  = rate_of_change(data['Close'])
    data['Momentum_14'] = compute_momentum(data['Close'])
    data['ATR_14']  = compute_atr(data)

    fwd_returns   = data['Close'].shift(-holding_days) / data['Close'] - 1
    signals       = get_signals(data)
    indicator_names = list(signals.keys())
    n_ind         = len(indicator_names)

    # ── Heatmap combinations ──
    # For each pair, compute stats based on ACTUALLY EXECUTED trades (no overlap)
    analysis_start  = end - timedelta(days=365 * years)
    data_trimmed    = data[data.index >= analysis_start].copy()
    spy_trimmed     = spy_raw[spy_raw.index >= analysis_start].copy()
    common_dates    = data_trimmed.index.intersection(spy_trimmed.index)
    data_trimmed    = data_trimmed.loc[common_dates]

    fwd_trim        = fwd_returns.reindex(common_dates)

    results = []
    for i in range(n_ind):
        for j in range(i + 1, n_ind):
            ind1 = indicator_names[i]
            ind2 = indicator_names[j]

            # Raw combined signal on trimmed window
            s1 = signals[ind1].reindex(common_dates).fillna(False)
            s2 = signals[ind2].reindex(common_dates).fillna(False)
            combined_raw = (s1 & s2).values.astype(bool)

            signals_total, trades_actual, skipped, trade_mask = \
                count_actual_trades(combined_raw, holding_days)

            # Win/loss based on actual entry days only
            entry_indices = np.where(trade_mask == 2)[0]
            if trades_actual == 0:
                avg_return = None; success_rate = None; win_count = 0; loss_count = 0
            else:
                entry_fwd = fwd_trim.iloc[entry_indices].dropna()
                if len(entry_fwd) == 0:
                    avg_return = None; success_rate = None; win_count = 0; loss_count = 0
                else:
                    avg_return   = float(round(entry_fwd.mean() * 100, 2))
                    win_count    = int((entry_fwd > 0).sum())
                    loss_count   = int((entry_fwd <= 0).sum())
                    success_rate = float(round(win_count / len(entry_fwd) * 100, 1))

            results.append({
                "indicator1":     ind1,
                "indicator2":     ind2,
                "signals_total":  signals_total,   # raw signal count
                "trades_actual":  trades_actual,   # actually executed
                "skipped":        skipped,         # blocked by overlap
                "win":            win_count,
                "loss":           loss_count,
                "success_rate":   success_rate,
                "avg_return":     avg_return,
            })

    valid      = [r for r in results if r["avg_return"] is not None]
    top_sorted = sorted(valid, key=lambda x: x["avg_return"], reverse=True)
    top3       = top_sorted[:3]
    top3_pairs = [(r["indicator1"], r["indicator2"]) for r in top3]
    for r in results:
        r["is_top"] = (r["indicator1"], r["indicator2"]) in top3_pairs

    # ── Equity curve ──
    spy_trimmed  = spy_raw[spy_raw.index >= analysis_start].copy()
    spy_trimmed  = spy_trimmed.loc[common_dates]

    ticker_daily = data_trimmed['Close'].pct_change().fillna(0)
    spy_daily    = spy_trimmed['Close'].pct_change().fillna(0)

    best  = top3[0]
    sig1  = signals[best["indicator1"]].reindex(common_dates).fillna(False)
    sig2  = signals[best["indicator2"]].reindex(common_dates).fillna(False)
    best_signal = sig1 & sig2

    equity_rows = build_equity_curve(
        signal=best_signal,
        ticker_daily_returns=ticker_daily,
        spy_daily_returns=spy_daily,
        holding_days=holding_days,
        dates=common_dates,
        ticker=ticker,
    )

    final = equity_rows[-1] if equity_rows else {}

    # Trade stats for best combo (recalculate cleanly)
    best_sig_arr = best_signal.values.astype(bool)
    sig_total, trades_act, skipped_ct, _ = count_actual_trades(best_sig_arr, holding_days)

    return {
        "ticker":       ticker,
        "holding_days": holding_days,
        "years":        years,
        "indicators":   indicator_names,
        "data_start":   common_dates[0].strftime('%Y-%m-%d'),
        "data_end":     common_dates[-1].strftime('%Y-%m-%d'),
        "combinations": results,
        "top3":         top3,
        "equity_curve": equity_rows,
        "equity_summary": {
            "strat_spy_final":  final.get("strat_spy",  INITIAL_CAPITAL),
            "strat_bond_final": final.get("strat_bond", INITIAL_CAPITAL),
            "buyhold_final":    final.get("buyhold",    INITIAL_CAPITAL),
            "spy_final":        final.get("spy",        INITIAL_CAPITAL),
            "bond_final":       final.get("bond",       INITIAL_CAPITAL),
            "best_combo":       f"{best['indicator1']} + {best['indicator2']}",
            "signals_total":    sig_total,
            "trades_actual":    trades_act,
            "skipped":          skipped_ct,
        },
    }
