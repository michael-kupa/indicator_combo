import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ─────────────────────────────────────────
# 지표 계산 함수들
# ─────────────────────────────────────────

def moving_average(data, window):
    return data.rolling(window=window).mean()

def ichimoku_cloud(data):
    nine_high = data['High'].rolling(9).max()
    nine_low  = data['Low'].rolling(9).min()
    conversion = (nine_high + nine_low) / 2

    twenty_six_high = data['High'].rolling(26).max()
    twenty_six_low  = data['Low'].rolling(26).min()
    base = (twenty_six_high + twenty_six_low) / 2

    span_a = ((conversion + base) / 2).shift(26)

    fifty_two_high = data['High'].rolling(52).max()
    fifty_two_low  = data['Low'].rolling(52).min()
    span_b = ((fifty_two_high + fifty_two_low) / 2).shift(26)

    lagging = data['Close'].shift(-26)
    return conversion, base, span_a, span_b, lagging

def envelopes(data, window=20, deviation=0.02):
    ma = data.rolling(window).mean()
    return ma * (1 + deviation), ma * (1 - deviation)

def exponential_moving_average(data, window):
    return data.ewm(span=window, adjust=False).mean()

def parabolic_sar(data, iaf=0.02, maxaf=0.2):
    length = len(data)
    high  = list(data['High'])
    low   = list(data['Low'])
    close = list(data['Close'])
    psar  = close[:]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af   = iaf
    hp   = high[0]
    lp   = low[0]

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
                if high[i] > hp:
                    hp = high[i]; af = min(af + iaf, maxaf)
                if low[i-1] < psar[i]: psar[i] = low[i-1]
                if low[i-2] < psar[i]: psar[i] = low[i-2]
            else:
                if low[i] < lp:
                    lp = low[i]; af = min(af + iaf, maxaf)
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
    macd_line   = data.ewm(span=fast, adjust=False).mean() - data.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

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
    high_max = data['High'].rolling(window).max()
    low_min  = data['Low'].rolling(window).min()
    return -100 * ((high_max - data['Close']) / (high_max - low_min))

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
# 신호 정의
# ─────────────────────────────────────────

def get_signals(data):
    return {
        'MA_20 (Trend Following)':          data['Close'] > data['MA_20'],
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
# 메인 분석 함수
# ─────────────────────────────────────────

def run_analysis(ticker: str, holding_days: int, years: int) -> dict:
    end   = datetime.today()
    start = end - timedelta(days=365 * years + 100)  # 지표 계산용 버퍼 포함

    data = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    # 컬럼 플래튼 (멀티인덱스 방지)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # 지표 추가
    data['MA_20']           = moving_average(data['Close'], 20)
    conv, base, spa, spb, lag = ichimoku_cloud(data)
    data['Conversion_Line'] = conv
    data['Base_Line']       = base
    data['Upper_Env'], data['Lower_Env'] = envelopes(data['Close'])
    data['EMA_26']          = exponential_moving_average(data['Close'], 26)
    psarbull, psarbear      = parabolic_sar(data)
    data['PSAR_Bull']       = psarbull
    data['RSI_14']          = compute_rsi(data['Close'], 14)
    data['RSI_3']           = compute_rsi(data['Close'], 3)
    data['MACD_Line'], data['Signal_Line'] = macd(data['Close'])
    data['OBV']             = on_balance_volume(data)
    data['Stoch_K'], data['Stoch_D'] = stochastic_oscillator(data)
    data['CCI_14']          = compute_cci(data)
    data['DI_Plus'], data['DI_Minus'], data['ADX'] = compute_dmi(data)
    data['WR_14']           = compute_wr(data)
    data['ROC_25']          = rate_of_change(data['Close'])
    data['Momentum_14']     = compute_momentum(data['Close'])
    data['ATR_14']          = compute_atr(data)

    # 수익률 계산 (holding_days 후)
    returns = data['Close'].shift(-holding_days) / data['Close'] - 1

    signals = get_signals(data)
    indicator_names = list(signals.keys())
    n = len(indicator_names)

    results = []

    for i in range(n):
        for j in range(i + 1, n):
            ind1 = indicator_names[i]
            ind2 = indicator_names[j]

            combined = signals[ind1] & signals[ind2]
            count    = int(combined.sum())

            if count == 0:
                avg_return   = None
                success_rate = None
                win_count    = 0
                loss_count   = 0
            else:
                r            = returns[combined].dropna()
                avg_return   = float(round(r.mean() * 100, 2))
                win_count    = int((r > 0).sum())
                loss_count   = int((r <= 0).sum())
                success_rate = float(round(win_count / len(r) * 100, 1)) if len(r) > 0 else None

            results.append({
                "indicator1":   ind1,
                "indicator2":   ind2,
                "count":        count,
                "win":          win_count,
                "loss":         loss_count,
                "success_rate": success_rate,   # % (예: 83.2)
                "avg_return":   avg_return,      # % (예: 2.1)
            })

    # 평균 수익률 기준 top 3
    valid   = [r for r in results if r["avg_return"] is not None]
    top3    = sorted(valid, key=lambda x: x["avg_return"], reverse=True)[:3]
    top3_pairs = [(r["indicator1"], r["indicator2"]) for r in top3]

    for r in results:
        r["is_top"] = (r["indicator1"], r["indicator2"]) in top3_pairs

    return {
        "ticker":        ticker,
        "holding_days":  holding_days,
        "years":         years,
        "indicators":    indicator_names,
        "data_start":    data.index[0].strftime('%Y-%m-%d'),
        "data_end":      data.index[-1].strftime('%Y-%m-%d'),
        "combinations":  results,
        "top3":          top3,
    }
