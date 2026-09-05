import logging
import traceback
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =============================================================================
# Configuration
# =============================================================================

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
QUICKCHART_URL = "https://quickchart.io/chart"
TELEGRAM_SEND_PHOTO_URL = "https://api.telegram.org/bot{}/sendPhoto"

REQUEST_TIMEOUT = 20
CANDLE_INTERVAL_MINUTES = 60
CANDLE_LIMIT = 50

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("ctos-chart-service")

# =============================================================================
# FastAPI
# =============================================================================

app = FastAPI(
    title="CTOS V1 Chart Service",
    description="Real candlestick chart generation and Telegram delivery service.",
    version="1.0.0",
)


# =============================================================================
# Request Model
# =============================================================================

class ChartRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    direction: str = Field(..., min_length=1)
    entry: float
    stop: float
    target: float
    gemini_verdict: str
    gemini_reasoning: str
    telegram_bot_token: str = Field(..., min_length=1)
    telegram_chat_id: str = Field(..., min_length=1)


# =============================================================================
# Utility Functions
# =============================================================================

def sanitize_ticker(ticker: str) -> str:
    """Normalize ticker symbols."""
    return ticker.strip().upper().replace("/", "").replace("-", "")


def calculate_rr(entry: float, stop: float, target: float) -> float:
    """Calculate reward-to-risk ratio."""
    risk = abs(entry - stop)
    reward = abs(target - entry)

    if risk <= 0:
        return 0.0

    return reward / risk


def format_price(value: float) -> str:
    """Format prices cleanly for chart labels and Telegram."""
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:,.4f}"
    return f"{value:.8f}"


def truncate_text(text: str, max_chars: int) -> str:
    """Trim text safely for Telegram/chart display."""
    text = (text or "").strip()

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3].rstrip() + "..."


# =============================================================================
# Kraken
# =============================================================================

def get_kraken_pair(ticker: str) -> str:
    """
    Map CTOS ticker symbols to Kraken pairs.

    Kraken symbols can vary depending on API version/listing, so the
    function includes the requested mappings plus common alternatives.
    """
    ticker = sanitize_ticker(ticker)

    mappings = {
        "BTCUSDC": "XBTUSD",
        "ETHUSDC": "ETHUSD",
        "BTCUSD": "XBTUSD",
        "ETHUSD": "ETHUSD",
        "BTCUSDT": "XBTUSD",
        "ETHUSDT": "ETHUSD",
    }

    return mappings.get(ticker, ticker)


def fetch_kraken_ohlcv(ticker: str) -> pd.DataFrame:
    """
    Fetch real OHLCV candles from Kraken public OHLC API.
    """
    pair = get_kraken_pair(ticker)

    logger.info(
        "Fetching Kraken OHLC data | ticker=%s | pair=%s | interval=%s | limit=%s",
        ticker,
        pair,
        CANDLE_INTERVAL_MINUTES,
        CANDLE_LIMIT,
    )

    response = requests.get(
        KRAKEN_OHLC_URL,
        params={
            "pair": pair,
            "interval": CANDLE_INTERVAL_MINUTES,
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(
            f"Kraken API returned errors: {payload.get('error')}"
        )

    result = payload.get("result", {})

    if not result:
        raise RuntimeError("Kraken returned an empty result.")

    # Kraken returns a dynamic pair key plus "last".
    pair_key = next(
        (key for key in result.keys() if key != "last"),
        None,
    )

    if not pair_key:
        raise RuntimeError("Could not find OHLC pair in Kraken response.")

    rows = result[pair_key]

    if not rows:
        raise RuntimeError("Kraken returned no OHLC candles.")

    # Kraken OHLC:
    # [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "count",
        ],
    )

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].dropna()

    df = df.tail(CANDLE_LIMIT).reset_index(drop=True)

    if len(df) < 5:
        raise RuntimeError(
            f"Kraken returned insufficient candles: {len(df)}"
        )

    # QuickChart/Chart.js expects JavaScript epoch milliseconds.
    df["timestamp_ms"] = (df["timestamp"] * 1000).astype("int64")

    logger.info(
        "Kraken successfully returned %d real candles for %s.",
        len(df),
        ticker,
    )

    return df


# =============================================================================
# Binance Fallback
# =============================================================================

def get_binance_symbol(ticker: str) -> str:
    """
    Convert requested ticker to Binance fallback symbol.

    Requirement specifies BTCUSDT as fallback. ETHUSDT is also supported.
    """
    ticker = sanitize_ticker(ticker)

    if ticker.startswith("BTC"):
        return "BTCUSDT"

    if ticker.startswith("ETH"):
        return "ETHUSDT"

    return "BTCUSDT"


def fetch_binance_ohlcv(ticker: str) -> pd.DataFrame:
    """
    Fetch real OHLCV candles from Binance public API.
    """
    symbol = get_binance_symbol(ticker)

    logger.info(
        "Fetching Binance fallback OHLC data | symbol=%s | interval=1h | limit=%s",
        symbol,
        CANDLE_LIMIT,
    )

    response = requests.get(
        BINANCE_KLINES_URL,
        params={
            "symbol": symbol,
            "interval": "1h",
            "limit": CANDLE_LIMIT,
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, list):
        raise RuntimeError(
            f"Unexpected Binance response: {payload}"
        )

    if len(payload) < 5:
        raise RuntimeError(
            f"Binance returned insufficient candles: {len(payload)}"
        )

    # Binance kline:
    # [
    #   open time,
    #   open,
    #   high,
    #   low,
    #   close,
    #   volume,
    #   close time,
    #   ...
    # ]
    rows = []

    for candle in payload:
        rows.append(
            {
                "timestamp": float(candle[0]) / 1000.0,
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
            }
        )

    df = pd.DataFrame(rows)

    df = df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].dropna()

    df = df.tail(CANDLE_LIMIT).reset_index(drop=True)

    df["timestamp_ms"] = (df["timestamp"] * 1000).astype("int64")

    logger.info(
        "Binance fallback successfully returned %d real candles.",
        len(df),
    )

    return df


# =============================================================================
# Dummy Data
# =============================================================================

def generate_dummy_ohlcv(ticker: str) -> pd.DataFrame:
    """
    Generate deterministic-looking fallback candles.

    This is used only if both public exchange APIs fail. The service logs
    this condition clearly so it is never mistaken for real market data.
    """
    logger.warning(
        "Generating DUMMY OHLCV data for %s because real exchange data "
        "could not be obtained.",
        ticker,
    )

    import random

    ticker = sanitize_ticker(ticker)

    if ticker.startswith("BTC"):
        base_price = 100000.0
    elif ticker.startswith("ETH"):
        base_price = 4000.0
    else:
        base_price = 100.0

    random.seed(int(time.time()))

    now = int(time.time())
    rows = []

    price = base_price

    for i in range(CANDLE_LIMIT):
        timestamp = now - (
            (CANDLE_LIMIT - i) * CANDLE_INTERVAL_MINUTES * 60
        )

        change = random.uniform(-0.012, 0.012)

        open_price = price
        close_price = price * (1 + change)

        high_price = max(open_price, close_price) * (
            1 + random.uniform(0.001, 0.008)
        )

        low_price = min(open_price, close_price) * (
            1 - random.uniform(0.001, 0.008)
        )

        volume = random.uniform(100, 1000)

        rows.append(
            {
                "timestamp": timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
        )

        price = close_price

    df = pd.DataFrame(rows)
    df["timestamp_ms"] = (df["timestamp"] * 1000).astype("int64")

    return df


# =============================================================================
# Market Data Orchestrator
# =============================================================================

def fetch_market_data(ticker: str) -> Tuple[pd.DataFrame, str]:
    """
    Try Kraken first, Binance second, and dummy data last.

    Returns:
        (DataFrame, data_source)
    """

    try:
        df = fetch_kraken_ohlcv(ticker)
        return df, "Kraken"
    except Exception:
        logger.error(
            "Kraken OHLC request failed:\n%s",
            traceback.format_exc(),
        )

    try:
        df = fetch_binance_ohlcv(ticker)
        return df, "Binance"
    except Exception:
        logger.error(
            "Binance fallback OHLC request failed:\n%s",
            traceback.format_exc(),
        )

    df = generate_dummy_ohlcv(ticker)

    return df, "DUMMY"


# =============================================================================
# QuickChart
# =============================================================================

def build_candlestick_chart_config(
    ticker: str,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    gemini_verdict: str,
    gemini_reasoning: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build QuickChart configuration using Chart.js financial/candlestick
    chart configuration.
    """

    rr = calculate_rr(entry, stop, target)

    title = (
        f"{ticker} | {direction} | R:R {rr:.2f}"
    )

    reasoning = truncate_text(
        gemini_reasoning,
        120,
    )

    # QuickChart's financial chart plugin expects:
    # {x: timestamp_ms, o: open, h: high, l: low, c: close}
    candle_data: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        candle_data.append(
            {
                "x": int(row["timestamp_ms"]),
                "o": float(row["open"]),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
            }
        )

    annotations = {
        # ---------------------------------------------------------------------
        # Entry
        # ---------------------------------------------------------------------
        "entryLine": {
            "type": "line",
            "yMin": entry,
            "yMax": entry,
            "borderColor": "#00BFFF",
            "borderWidth": 2,
            "borderDash": [8, 6],
            "label": {
                "display": True,
                "content": f"Entry: ${format_price(entry)}",
                "position": "start",
                "backgroundColor": "#00BFFF",
                "color": "#FFFFFF",
                "font": {
                    "size": 11,
                    "weight": "bold",
                },
            },
        },

        # ---------------------------------------------------------------------
        # Stop Loss
        # ---------------------------------------------------------------------
        "stopLine": {
            "type": "line",
            "yMin": stop,
            "yMax": stop,
            "borderColor": "#FF4444",
            "borderWidth": 2,
            "borderDash": [8, 6],
            "label": {
                "display": True,
                "content": f"Stop: ${format_price(stop)}",
                "position": "start",
                "backgroundColor": "#FF4444",
                "color": "#FFFFFF",
                "font": {
                    "size": 11,
                    "weight": "bold",
                },
            },
        },

        # ---------------------------------------------------------------------
        # Target
        # ---------------------------------------------------------------------
        "targetLine": {
            "type": "line",
            "yMin": target,
            "yMax": target,
            "borderColor": "#44FF88",
            "borderWidth": 2,
            "borderDash": [8, 6],
            "label": {
                "display": True,
                "content": f"Target: ${format_price(target)}",
                "position": "start",
                "backgroundColor": "#44FF88",
                "color": "#000000",
                "font": {
                    "size": 11,
                    "weight": "bold",
                },
            },
        },

        # ---------------------------------------------------------------------
        # Gemini reasoning box
        # ---------------------------------------------------------------------
        "reasoningBox": {
            "type": "label",
            "xValue": int(df["timestamp_ms"].iloc[-1]),
            "yValue": float(df["high"].max()),
            "content": [
                "GEMINI",
                reasoning,
            ],
            "backgroundColor": "rgba(35, 35, 60, 0.92)",
            "borderColor": "#FFFFFF",
            "borderWidth": 1,
            "borderRadius": 6,
            "color": "#FFFFFF",
            "padding": 10,
            "textAlign": "left",
            "font": [
                {
                    "size": 12,
                    "weight": "bold",
                },
                {
                    "size": 10,
                    "weight": "normal",
                },
            ],
            "position": "center",
        },
    }

    config = {
        "type": "candlestick",

        "data": {
            "datasets": [
                {
                    "label": ticker,
                    "data": candle_data,

                    # Candlestick colors
                    "borderColor": "#FFFFFF",
                    "color": {
                        "up": "#44FF88",
                        "down": "#FF4444",
                        "unchanged": "#AAAAAA",
                    },

                    "barThickness": "flex",
                    "maxBarThickness": 14,
                }
            ]
        },

        "options": {
            "responsive": True,

            "plugins": {
                "legend": {
                    "display": False,
                },

                "title": {
                    "display": True,
                    "text": title,
                    "color": "#FFFFFF",
                    "font": {
                        "size": 18,
                        "weight": "bold",
                    },
                    "padding": {
                        "top": 10,
                        "bottom": 15,
                    },
                },

                "annotation": {
                    "annotations": annotations,
                },
            },

            "scales": {
                "x": {
                    "type": "time",

                    "time": {
                        "unit": "hour",
                        "displayFormats": {
                            "hour": "dd MMM HH:mm",
                        },
                    },

                    "ticks": {
                        "color": "#FFFFFF",
                        "maxTicksLimit": 10,
                    },

                    "grid": {
                        "color": "rgba(255,255,255,0.08)",
                    },

                    "title": {
                        "display": True,
                        "text": "Time",
                        "color": "#FFFFFF",
                    },
                },

                "y": {
                    "position": "right",

                    "ticks": {
                        "color": "#FFFFFF",
                    },

                    "grid": {
                        "color": "rgba(255,255,255,0.08)",
                    },

                    "title": {
                        "display": True,
                        "text": "Price",
                        "color": "#FFFFFF",
                    },
                },
            },

            "layout": {
                "padding": {
                    "top": 20,
                    "right": 20,
                    "bottom": 15,
                    "left": 10,
                }
            },
        },
    }

    return config


def generate_quickchart_image(
    ticker: str,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    gemini_verdict: str,
    gemini_reasoning: str,
    df: pd.DataFrame,
) -> bytes:
    """
    Generate a PNG candlestick chart using QuickChart POST API.
    """

    logger.info(
        "Generating QuickChart candlestick image | ticker=%s | candles=%d",
        ticker,
        len(df),
    )

    config = build_candlestick_chart_config(
        ticker=ticker,
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        gemini_verdict=gemini_verdict,
        gemini_reasoning=gemini_reasoning,
        df=df,
    )

    payload = {
        "chart": config,
        "width": 1200,
        "height": 700,
        "format": "png",
        "backgroundColor": "#1A1A2E",
        "version": "4",
    }

    response = requests.post(
        QUICKCHART_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    content_type = response.headers.get("content-type", "")

    if "image" not in content_type:
        logger.error(
            "QuickChart returned unexpected content type: %s",
            content_type,
        )

        raise RuntimeError(
            "QuickChart did not return an image."
        )

    if not response.content:
        raise RuntimeError(
            "QuickChart returned an empty image."
        )

    logger.info(
        "QuickChart generated image successfully | bytes=%d",
        len(response.content),
    )

    return response.content


# =============================================================================
# Telegram
# =============================================================================

def build_telegram_caption(
    ticker: str,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    gemini_verdict: str,
    gemini_reasoning: str,
) -> str:
    """
    Build Telegram caption.
    """

    rr = calculate_rr(entry, stop, target)

    reasoning = truncate_text(
        gemini_reasoning,
        200,
    )

    caption = (
        "📊 CTOS V1 TRADE ANALYSIS\n"
        "\n"
        f"Ticker: {ticker}\n"
        f"Direction: {direction}\n"
        f"Entry: ${format_price(entry)}\n"
        f"Stop: ${format_price(stop)}\n"
        f"Target: ${format_price(target)}\n"
        f"R:R: 1:{rr:.2f}\n"
        "\n"
        f"🤖 AI Verdict: {gemini_verdict}\n"
        "\n"
        f"🧠 Reasoning: {reasoning}"
    )

    return caption


def send_to_telegram(
    bot_token: str,
    chat_id: str,
    image_bytes: bytes,
    caption: str,
) -> Dict[str, Any]:
    """
    Send PNG chart directly to Telegram using sendPhoto multipart/form-data.
    """

    url = TELEGRAM_SEND_PHOTO_URL.format(bot_token)

    logger.info(
        "Sending chart to Telegram | chat_id=%s | image_bytes=%d",
        chat_id,
        len(image_bytes),
    )

    files = {
        "photo": (
            "ctos_chart.png",
            image_bytes,
            "image/png",
        )
    }

    data = {
        "chat_id": chat_id,
        "caption": caption,
    }

    response = requests.post(
        url,
        data=data,
        files=files,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(
            f"Telegram API returned failure: {result}"
        )

    logger.info(
        "Telegram chart sent successfully | chat_id=%s",
        chat_id,
    )

    return result


# =============================================================================
# Health Check
# =============================================================================

@app.get("/")
def root() -> Dict[str, str]:
    return {
        "status": "success",
        "service": "CTOS V1 Chart Service",
        "version": "1.0.0",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "success",
    }


# =============================================================================
# Main Endpoint
# =============================================================================

@app.post("/generate-chart")
def generate_chart(request: ChartRequest) -> Dict[str, Any]:
    """
    Main CTOS V1 endpoint.

    Flow:
        Google Apps Script
            ↓
        FastAPI
            ↓
        Kraken OHLC
            ↓
        Binance fallback
            ↓
        Dummy fallback
            ↓
        QuickChart candlestick
            ↓
        Telegram sendPhoto
    """

    try:
        logger.info("=" * 80)
        logger.info("CTOS V1 /generate-chart request received")
        logger.info("Ticker: %s", request.ticker)
        logger.info("Direction: %s", request.direction)
        logger.info("Entry: %s", request.entry)
        logger.info("Stop: %s", request.stop)
        logger.info("Target: %s", request.target)
        logger.info("Gemini verdict: %s", request.gemini_verdict)
        logger.info("=" * 80)

        ticker = sanitize_ticker(request.ticker)
        direction = request.direction.strip()

        # ---------------------------------------------------------------------
        # Basic validation
        # ---------------------------------------------------------------------

        if direction.lower() not in {"long", "short"}:
            raise ValueError(
                "direction must be either 'Long' or 'Short'."
            )

        if request.entry <= 0:
            raise ValueError("entry must be greater than zero.")

        if request.stop <= 0:
            raise ValueError("stop must be greater than zero.")

        if request.target <= 0:
            raise ValueError("target must be greater than zero.")

        if abs(request.entry - request.stop) <= 0:
            raise ValueError(
                "entry and stop cannot be identical."
            )

        # ---------------------------------------------------------------------
        # Fetch real market data
        # ---------------------------------------------------------------------

        df, data_source = fetch_market_data(ticker)

        logger.info(
            "Market data source selected: %s",
            data_source,
        )

        # ---------------------------------------------------------------------
        # Generate chart
        # ---------------------------------------------------------------------

        try:
            chart_image = generate_quickchart_image(
                ticker=ticker,
                direction=direction,
                entry=request.entry,
                stop=request.stop,
                target=request.target,
                gemini_verdict=request.gemini_verdict,
                gemini_reasoning=request.gemini_reasoning,
                df=df,
            )

        except Exception:
            logger.error(
                "Chart generation failed:\n%s",
                traceback.format_exc(),
            )

            # Explicitly propagate the error so FastAPI returns HTTP 500.
            raise

        # ---------------------------------------------------------------------
        # Telegram caption
        # ---------------------------------------------------------------------

        caption = build_telegram_caption(
            ticker=ticker,
            direction=direction,
            entry=request.entry,
            stop=request.stop,
            target=request.target,
            gemini_verdict=request.gemini_verdict,
            gemini_reasoning=request.gemini_reasoning,
        )

        # ---------------------------------------------------------------------
        # Telegram delivery
        # ---------------------------------------------------------------------

        telegram_result = send_to_telegram(
            bot_token=request.telegram_bot_token,
            chat_id=request.telegram_chat_id,
            image_bytes=chart_image,
            caption=caption,
        )

        rr = calculate_rr(
            request.entry,
            request.stop,
            request.target,
        )

        logger.info(
            "CTOS V1 chart generation completed successfully."
        )

        return {
            "status": "success",
            "ticker": ticker,
            "direction": direction,
            "data_source": data_source,
            "candles": len(df),
            "rr": round(rr, 4),
            "telegram_sent": bool(
                telegram_result.get("ok")
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        error_traceback = traceback.format_exc()

        logger.error(
            "CTOS V1 /generate-chart failed:\n%s",
            error_traceback,
        )

        # Return traceback in JSON so Render logs/debugging clearly show it.
        return {
            "status": "error",
            "error": str(exc),
            "traceback": error_traceback,
        }


# =============================================================================
# Local Development Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
