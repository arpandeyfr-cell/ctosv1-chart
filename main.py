"""
CTOS V1 - CHART ENGINE
FastAPI microservice for generating candlestick charts with AI reasoning.
Deploys on Render (free tier).
"""

import io
import logging
import requests
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ===================== CONFIGURATION =====================
app = FastAPI(title="CTOS V1 Chart Engine", version="1.0.0")

# ===================== DATA MODELS =====================

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

# ===================== CHART GENERATION =====================

def generate_chart_image(ticker, entry, stop, target, direction, gemini_reasoning):
    df = fetch_kraken_ohlcv(ticker, interval_minutes=60, limit=50)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_df = pd.DataFrame({'RSI': rsi})
    
    mc = mpf.make_marketcolors(
        up='#2ECC71', down='#E74C3C',
        wick={'up':'#2ECC71', 'down':'#E74C3C'},
        edge='#1A1A2E',
        volume={'up':'#2ECC71', 'down':'#E74C3C'}
    )
    s = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mc, gridcolor='#444444', figcolor='#1A1A2E')
    
    fig, axes = mpf.plot(df, 
                         type='candle', 
                         style=s,
                         volume=True,
                         hlines=dict(hlines=[entry, stop, target], colors=['#00BFFF', '#FF4444', '#44FF88'], linestyle='--'),
                         figsize=(12, 8),
                         returnfig=True,
                         addplot=mpf.make_addplot(rsi, panel=1, color='#FFB347', ylabel='RSI'),
                         panel_ratios=(2, 1, 0.5))
    
    ax = axes[0]
    
    ax.text(0.02, 0.98, f'Entry: ${entry:.0f}', transform=ax.transAxes, color='#00BFFF', fontsize=10, va='top')
    ax.text(0.02, 0.92, f'Stop: ${stop:.0f}', transform=ax.transAxes, color='#FF4444', fontsize=10, va='top')
    ax.text(0.02, 0.86, f'Target: ${target:.0f}', transform=ax.transAxes, color='#44FF88', fontsize=10, va='top')
    
    props = dict(boxstyle='round', facecolor='#2D2D44', alpha=0.85)
    ax.text(0.65, 0.98, f'🧠 AI Verdict:\n{gemini_reasoning[:100]}...', transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', color='#FFFFFF', bbox=props)
    
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
        
        result = send_telegram_image(
            image_bytes,
            caption,
            request.telegram_bot_token,
            request.telegram_chat_id
        )
        
        if result.get('ok'):
            return {"status": "success", "message": "Chart sent to Telegram"}
        else:
            return {"status": "error", "message": result.get('description', 'Unknown error')}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "CTOS V1 Chart Engine is running. Use /docs for API documentation."}
