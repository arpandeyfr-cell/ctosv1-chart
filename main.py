import io
import logging
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
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

# ===================== GENERATE CANDLESTICK CHART (via QuickChart POST) =====================

def generate_candlestick_image(ticker, entry, stop, target, direction, reasoning):
    """
    Generates a candlestick chart using QuickChart POST method.
    Returns binary image data.
    """
    # Build candlestick data (dummy data for now, but we can fetch real data later)
    # QuickChart candlestick format: each candle is {o: open, h: high, l: low, c: close}
    candles = [
        {"o": 79000, "h": 79100, "l": 78900, "c": 79050},
        {"o": 79050, "h": 79200, "l": 79000, "c": 79150},
        {"o": 79150, "h": 79300, "l": 79100, "c": 79200},
        {"o": 79200, "h": 79400, "l": 79150, "c": 79350},
        {"o": 79350, "h": 79500, "l": 79300, "c": 79400},
        {"o": 79400, "h": 79550, "l": 79350, "c": 79450},
        {"o": 79450, "h": 79600, "l": 79400, "c": 79500},
        {"o": 79500, "h": 79700, "l": 79450, "c": 79600},
        {"o": 79600, "h": 79800, "l": 79550, "c": 79700},
        {"o": 79700, "h": 79900, "l": 79650, "c": 79800},
        {"o": 79800, "h": 80000, "l": 79750, "c": 79900},
        {"o": 79900, "h": 80050, "l": 79850, "c": 79950},
        {"o": 79950, "h": 80100, "l": 79900, "c": 80000},
        {"o": 80000, "h": 80200, "l": 79950, "c": 80100},
        {"o": 80100, "h": 80300, "l": 80050, "c": 80200}
    ]
    
    # Build QuickChart candlestick config
    chart_config = {
        "type": "candlestick",
        "data": {
            "labels": [f"Day {i+1}" for i in range(len(candles))],
            "datasets": [
                {
                    "label": "Price",
                    "data": candles,
                    "borderColor": "#FFB347",
                    "backgroundColor": "rgba(255, 179, 71, 0.1)"
                }
            ]
        },
        "options": {
            "plugins": {
                "title": {
                    "display": True,
                    "text": f"{ticker} | {direction} | Entry: ${entry:.0f} | Stop: ${stop:.0f} | Target: ${target:.0f}",
                    "color": "#FFFFFF",
                    "font": {"size": 16, "weight": "bold"}
                },
                "legend": {
                    "labels": {"color": "#FFFFFF"}
                },
                "annotation": {
                    "annotations": {
                        "entryLine": {
                            "type": "line",
                            "yMin": entry,
                            "yMax": entry,
                            "borderColor": "#00BFFF",
                            "borderWidth": 2,
                            "borderDash": [5, 5],
                            "label": {
                                "content": f"Entry: ${entry:.0f}",
                                "enabled": True,
                                "position": "start",
                                "font": {"size": 10, "weight": "bold"},
                                "color": "#00BFFF"
                            }
                        },
                        "stopLine": {
                            "type": "line",
                            "yMin": stop,
                            "yMax": stop,
                            "borderColor": "#FF4444",
                            "borderWidth": 2,
                            "borderDash": [5, 5],
                            "label": {
                                "content": f"Stop: ${stop:.0f}",
                                "enabled": True,
                                "position": "start",
                                "font": {"size": 10, "weight": "bold"},
                                "color": "#FF4444"
                            }
                        },
                        "targetLine": {
                            "type": "line",
                            "yMin": target,
                            "yMax": target,
                            "borderColor": "#44FF88",
                            "borderWidth": 2,
                            "borderDash": [5, 5],
                            "label": {
                                "content": f"Target: ${target:.0f}",
                                "enabled": True,
                                "position": "start",
                                "font": {"size": 10, "weight": "bold"},
                                "color": "#44FF88"
                            }
                        }
                    }
                }
            },
            "scales": {
                "x": {
                    "ticks": {"color": "#FFFFFF"},
                    "grid": {"color": "#333333"}
                },
                "y": {
                    "ticks": {"color": "#FFFFFF"},
                    "grid": {"color": "#333333"}
                }
            }
        }
    }
    
    # Use QuickChart POST method
    payload = {
        "chart": chart_config,
        "backgroundColor": "#1A1A2E",
        "width": 800,
        "height": 500,
        "format": "png"
    }
    
    response = requests.post(
        "https://quickchart.io/chart",
        json=payload,
        timeout=15
    )
    
    if response.status_code != 200:
        logger.error(f"QuickChart error: {response.status_code}")
        logger.error(f"QuickChart response: {response.text[:200]}")
        raise Exception(f"QuickChart error: {response.status_code}")
    
    return response.content

# ===================== SEND TO TELEGRAM =====================

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
        logger.info(f"Generating candlestick chart for {request.ticker}")
        
        # 1. Generate candlestick chart image
        image_bytes = generate_candlestick_image(
            request.ticker,
            request.entry,
            request.stop,
            request.target,
            request.direction,
            request.gemini_reasoning
        )
        
        # 2. Build caption
        caption = (
            f"🚀 <b>NEW POSITION</b>\n\n"
            f"Ticker: {request.ticker}\n"
            f"Direction: {request.direction}\n"
            f"Entry: ${request.entry:.0f}\n"
            f"Stop: ${request.stop:.0f}\n"
            f"Target: ${request.target:.0f}\n\n"
            f"🧠 <b>AI Verdict:</b> {request.gemini_verdict}\n"
            f"📝 <b>Reasoning:</b> {request.gemini_reasoning[:300]}..."
        )
        
        # 3. Send to Telegram
        result = send_telegram_image(image_bytes, caption, request.telegram_bot_token, request.telegram_chat_id)
        
        if result.get('ok'):
            return {"status": "success", "message": "Candlestick chart sent to Telegram"}
        else:
            logger.error(f"Telegram error: {result}")
            return {"status": "error", "message": result.get('description', 'Unknown error')}
            
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "CTOS V1 Chart Engine is running. Use /docs for API documentation."}
