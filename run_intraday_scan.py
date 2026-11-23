python3 run_intraday_scan.py
#!/usr/bin/env python3
"""
Multi-Confluence Scanner (KAMA + CRSI + ATR + Time-Scaled Volume)
Sends Telegram inline BUY and SKIP alerts with Entry, SL, and Target for each hit.
"""

import datetime
import json
import os
from typing import Any
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
NSE_OPEN = datetime.time(9, 15)
NSE_CLOSE = datetime.time(15, 30)
SESSION_MINUTES = 375.0

WATCHLIST = [
    "SUNPHARMA.NS", "CIPLA.NS", "SBIN.NS", "MARUTI.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "RELIANCE.NS", "TCS.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "LT.NS",
]


def elapsed_session_fraction(now_ist: datetime.datetime | None = None) -> float:
    """
    Returns fraction of the 09:15-15:30 IST session elapsed.
    Returns 1.0 outside market hours or after close.
    """
    dt = now_ist or datetime.datetime.now(IST)
    t = dt.time()
    if t <= NSE_OPEN:
        return 0.05
    if t >= NSE_CLOSE:
        return 1.0
    elapsed = (t.hour * 60 + t.minute) - (NSE_OPEN.hour * 60 + NSE_OPEN.minute)
    return max(min(elapsed / SESSION_MINUTES, 1.0), 0.05)


def compute_kama(close: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    change = close.diff(period).abs()
    volatility = close.diff().abs().rolling(period).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0.0)
    sc = ((er * (2.0 / (fast + 1) - 2.0 / (slow + 1))) + (2.0 / (slow + 1))) ** 2

    kama = pd.Series(np.nan, index=close.index, dtype=float)
    if len(close) >= period:
        kama.iloc[period - 1] = float(close.iloc[period - 1])
        for i in range(period, len(close)):
            kama.iloc[i] = float(kama.iloc[i - 1]) + float(sc.iloc[i]) * (
                float(close.iloc[i]) - float(kama.iloc[i - 1])
            )
    return kama


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_connors_rsi(close: pd.Series) -> pd.Series:
    rsi_3 = compute_rsi(close, 3)
    direction = np.sign(close.diff()).fillna(0.0)
    streak, curr = [], 0
    for d in direction:
        if d > 0:
            curr = curr + 1 if curr > 0 else 1
        elif d < 0:
            curr = curr - 1 if curr < 0 else -1
        else:
            curr = 0
        streak.append(curr)
    streak_rsi = compute_rsi(pd.Series(streak, index=close.index, dtype=float), 2)
    pct_rank = close.pct_change().rolling(100).apply(
        lambda x: (x[:-1] < x[-1]).mean() * 100.0,
        raw=True,
    )
    return (rsi_3 + streak_rsi + pct_rank) / 3.0


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def fetch_sharekhan_ltp_map(symbols: list[str], client: Any | None = None) -> dict[str, dict[str, float]]:
    """
    Optional live quote hook for Sharekhan ShareConnect.
    Return { "SBIN.NS": {"ltp": 1153.85, "volume": 4200000} } when connected.
    """
    _ = (symbols, client)
    return {}


def build_callback_payload(payload: dict[str, Any]) -> str:
    """Serializes compact JSON and enforces Telegram's 64-byte callback_data limit."""
    encoded = json.dumps(payload, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 64:
        raise ValueError(f"callback_data exceeds 64 bytes ({len(encoded.encode('utf-8'))}): {encoded}")
    return encoded


def send_telegram_hit_alert(hit: dict[str, Any], scan_time_ist: str) -> bool:
    """
    Sends an inline BUY / SKIP alert card for a single matching symbol.
    Returns True if delivered, False if Telegram credentials are missing or call fails.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        print(f"[SKIP ALERT] {hit['Symbol']}: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        return False

    sym = str(hit["Symbol"])
    entry = float(hit["LTP (₹)"])
    sl = float(hit["Stop-Loss (-1.1 ATR)"])
    tgt = float(hit["Target (+1.8 ATR)"])
    kama = float(hit["KAMA (₹)"])
    crsi = float(hit["CRSI"])

    buy_data = build_callback_payload({
        "a": "B",
        "s": sym,
        "p": round(entry, 2),
        "l": round(sl, 2),
        "t": round(tgt, 2),
    })
    skip_data = build_callback_payload({
        "a": "S",
        "s": sym,
    })

    message_text = (
        f"📈 *SWING SIGNAL — NSE:{sym}*\n"
        f"• *Entry (LTP):* ₹{entry:,.2f}\n"
        f"• *Target (+1.8 ATR):* ₹{tgt:,.2f}\n"
        f"• *Stop-Loss (-1.1 ATR):* ₹{sl:,.2f}\n"
        f"• *KAMA:* ₹{kama:,.2f} | *CRSI:* {crsi:.1f}\n"
        f"• *Time:* {scan_time_ist}"
    )

    reply_markup = {
        "inline_keyboard": [
            [
                {"text": f"✅ BUY {sym} @ ₹{entry:,.2f}", "callback_data": buy_data},
                {"text": "⏭️ SKIP", "callback_data": skip_data},
            ]
        ]
    }

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message_text,
                "parse_mode": "Markdown",
                "reply_markup": reply_markup,
            },
            timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        if not body.get("ok"):
            print(f"[ERROR ALERT] {sym}: Telegram API returned {body}")
            return False
        msg_id = body.get("result", {}).get("message_id")
        print(f"[SENT ALERT] {sym} -> message_id={msg_id}")
        return True
    except Exception as exc:
        print(f"[ERROR ALERT] {sym}: {exc}")
        return False


def run_intraday_scan(sharekhan_client: Any | None = None, send_alerts: bool = True) -> list[dict[str, Any]]:
    now_ist = datetime.datetime.now(IST)
    scan_time_str = now_ist.strftime("%Y-%m-%d %H:%M:%S %Z")
    vol_scale = elapsed_session_fraction(now_ist)
    print(f"[{scan_time_str}] Running Multi-Confluence Scan (vol_scale={vol_scale:.2f})...")

    data = yf.download(
        WATCHLIST,
        period="6mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    live_quotes = fetch_sharekhan_ltp_map(WATCHLIST, sharekhan_client)
    hits: list[dict[str, Any]] = []

    for sym in WATCHLIST:
        if isinstance(data.columns, pd.MultiIndex):
            if sym not in data.columns.get_level_values(0):
                continue
            df = data[sym].dropna(how="all").copy()
        else:
            df = data.dropna(how="all").copy()

        df = df.dropna(subset=["High", "Low", "Close", "Volume"])
        if len(df) < 105:
            continue

        kama = compute_kama(df["Close"])
        kama_slope = kama.diff()
        crsi = compute_connors_rsi(df["Close"])
        atr = compute_atr(df, 14)
        vol_sma20 = df["Volume"].rolling(20).mean()

        quote = live_quotes.get(sym, {})
        ltp = float(quote.get("ltp", df["Close"].iloc[-1]))
        curr_vol = float(quote.get("volume", df["Volume"].iloc[-1]))

        last_kama = float(kama.iloc[-1])
        last_slope = float(kama_slope.iloc[-1])
        last_crsi = float(crsi.iloc[-1])
        prev_crsi = float(crsi.iloc[-2])
        last_atr = float(atr.iloc[-1])
        scaled_vol_req = float(vol_sma20.iloc[-1]) * vol_scale

        if any(np.isnan(v) for v in (ltp, last_kama, last_slope, last_crsi, prev_crsi, last_atr, scaled_vol_req)):
            continue

        cond_trend = (ltp > last_kama) and (last_slope > 0.0)
        cond_crsi = (last_crsi < 15.0) or (prev_crsi < 15.0 and last_crsi < 25.0)
        cond_vol = curr_vol > scaled_vol_req

        if cond_trend and cond_crsi and cond_vol:
            hit = {
                "Symbol": sym.replace(".NS", ""),
                "LTP (₹)": round(ltp, 2),
                "KAMA (₹)": round(last_kama, 2),
                "CRSI": round(last_crsi, 2),
                "Target (+1.8 ATR)": round(ltp + 1.8 * last_atr, 2),
                "Stop-Loss (-1.1 ATR)": round(ltp - 1.1 * last_atr, 2),
            }
            hits.append(hit)
            if send_alerts:
                send_telegram_hit_alert(hit, scan_time_str)

    if hits:
        print(pd.DataFrame(hits).to_string(index=False))
    else:
        print("No symbols meet all KAMA + CRSI + scaled volume criteria right now.")

    return hits

python3 - <<'PY'
from run_intraday_scan import run_intraday_scan
print(run_intraday_scan(send_alerts=False))
PY


if __name__ == "__main__":
    run_intraday_scan()
