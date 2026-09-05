import io
import logging
import traceback
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ===================== LOGGING SETUP =====================
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="CTOS V1 Chart Engine", version="1.0.0")

class TradeRequest(BaseModel):
    ticker: str
    direction: str
    entry: float
    stop: float
    target: float
    gemini_verdict: str
    gemini_reasoning: str
    telegram_bot_token: str
    telegram_chat_id: str

# ===================== KRAKEN DATA FETCH =====================
def fetch_kraken_ohlcv(pair, interval_minutes=60, limit=50):
    symbol_map = {
        "BTCUSDC": "XBTUSD",
        "BTCUSDT": "XBTUSD",
        "ETHUSDC": "ETHUSD",
        "ETHUSDT": "ETHUSD",
        "SOLUSDC": "SOLUSD",
        "XRPUSDC": "XRPUSD"
    }
    kraken_pair = symbol_map.get(pair, "XBTUSD")
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        "pair": kraken_pair,
        "interval": interval_minutes,
        "since": int((datetime.now() - timedelta(days=7)).timestamp())
    }
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    if data.get("error"):
        raise Exception(f"Kraken API error: {data['error']}")
    ohlc_data = data["result"][kraken_pair]
    df = pd.DataFrame(ohlc_data, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df.astype(float)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df.tail(limit)

# ===================== CHART GENERATION (pure matplotlib) =====================
def generate_chart_image(ticker, entry, stop, target, direction, gemini_reasoning):
    logger.debug("Generating chart for %s", ticker)
    # 1. Try to fetch real Kraken data
    try:
        df = fetch_kraken_ohlcv(ticker)
        logger.debug("Kraken data fetched: %d candles", len(df))
    except Exception as e:
        logger.warning("Kraken fetch failed: %s. Using dummy data.", e)
        # Generate dummy data around the entry price
        base_price = entry
        times = [datetime.now() - timedelta(hours=i) for i in range(50, 0, -1)]
        df = pd.DataFrame({
            'time': times,
            'open': [base_price + (i * 0.01) for i in range(50)],
            'high': [base_price + (i * 0.02) + 1 for i in range(50)],
            'low': [base_price + (i * 0.01) - 1 for i in range(50)],
            'close': [base_price + (i * 0.015) for i in range(50)],
            'volume': [100 + i * 0.5 for i in range(50)]
        })
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    
    # 2. Create the chart
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1A1A2E')
    ax.set_facecolor('#1A1A2E')
    
    # Candlesticks using matplotlib's candlestick2_ohlc (mplfinance would be easier, but we avoid extra dependencies)
    # We'll use a simple line chart for price (since pure matplotlib candlestick is verbose)
    ax.plot(df.index, df['close'], color='#FFB347', linewidth=2, label='Price (Close)')
    ax.fill_between(df.index, df['close'], df['close'].min(), color='#FFB347', alpha=0.1)
    
    # Entry, Stop, Target lines
    ax.axhline(y=entry, color='#00BFFF', linestyle='--', linewidth=1.5, label=f'Entry: ${entry:.0f}')
    ax.axhline(y=stop, color='#FF4444', linestyle='--', linewidth=1.5, label=f'Stop: ${stop:.0f}')
    ax.axhline(y=target, color='#44FF88', linestyle='--', linewidth=1.5, label=f'Target: ${target:.0f}')
    
    # Labels and styling
    ax.set_title(f'{ticker} | {direction} | Risk: ${abs(entry-stop):.0f} pts | R:R: {(abs(target-entry)/abs(stop-entry)):.2f}:1',
                 color='#FFFFFF', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price', color='#FFFFFF')
    ax.tick_params(colors='#FFFFFF')
    ax.grid(color='#444444', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', facecolor='#2D2D44', edgecolor='#444444', labelcolor='#FFFFFF')
    
    # Gemini reasoning text box
    props = dict(boxstyle='round', facecolor='#2D2D44', alpha=0.85)
    ax.text(0.65, 0.05, f'🧠 AI Verdict:\n{gemini_reasoning[:120]}...', transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', color='#FFFFFF', bbox=props)
    
    plt.tight_layout()
    
    # Save to buffer
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png', bbox_inches='tight', facecolor='#1A1A2E')
    img_data.seek(0)
    plt.close(fig)
    return img_data

# ===================== TELEGRAM SENDER =====================
def send_telegram_image(image_bytes, caption, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {'photo': ('chart.png', image_bytes, 'image/png')}
    data = {'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'}
    response = requests.post(url, files=files, data=data, timeout=15)
    return response.json()

# ===================== FASTAPI ENDPOINT =====================
@app.post("/generate-chart")
async def generate_chart(request: TradeRequest):
    try:
        logger.debug("Received request for %s", request.ticker)
        image_bytes = generate_chart_image(
            request.ticker,
            request.entry,
            request.stop,
            request.target,
            request.direction,
            request.gemini_reasoning
        )
        caption = (
            f"🚀 <b>NEW POSITION</b>\n\n"
            f"Ticker: {request.ticker}\n"
            f"Direction: {request.direction}\n"
            f"Entry: ${request.entry}\n"
            f"Stop: ${request.stop}\n"
            f"Target: ${request.target}\n\n"
            f"🧠 <b>AI Verdict:</b> {request.gemini_verdict}\n"
            f"📝 <b>Reasoning:</b> {request.gemini_reasoning[:200]}..."
        )
        result = send_telegram_image(image_bytes, caption, request.telegram_bot_token, request.telegram_chat_id)
        if result.get('ok'):
            return {"status": "success", "message": "Chart sent to Telegram"}
        else:
            return {"status": "error", "message": result.get('description', 'Unknown error')}
    except Exception as e:
        logger.error("Error in generate_chart: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "CTOS V1 Chart Engine is running. Use /docs for API documentation."}
