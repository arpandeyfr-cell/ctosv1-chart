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

# ===================== GENERATE CHART (via QuickChart POST) =====================

def generate_chart_image(ticker, entry, stop, target, direction, reasoning):
    """
    Generates a chart image using QuickChart's POST method.
    Returns a binary image blob.
    """
    # Build a simple line chart with entry/stop/target levels
    chart_config = {
        "type": "line",
        "data": {
            "labels": list(range(1, 21)),
            "datasets": [
                {
                    "label": "Price",
                    "data": [entry - 50 + i * 10 for i in range(20)],
                    "borderColor": "#FFB347",
                    "fill": False
                },
                {
                    "label": f"Entry: ${entry:.0f}",
                    "data": [entry] * 20,
                    "borderColor": "#00BFFF",
                    "borderDash": [5, 5]
                },
                {
                    "label": f"Stop: ${stop:.0f}",
                    "data": [stop] * 20,
                    "borderColor": "#FF4444",
                    "borderDash": [5, 5]
                },
                {
                    "label": f"Target: ${target:.0f}",
                    "data": [target] * 20,
                    "borderColor": "#44FF88",
                    "borderDash": [5, 5]
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
                }
            },
            "scales": {
                "x": {
                    "ticks": {"color": "#FFFFFF", "display": False},
                    "grid": {"color": "#333333"}
                },
                "y": {
                    "ticks": {"color": "#FFFFFF"},
                    "grid": {"color": "#333333"}
                }
            }
        }
    }
    
    # Use QuickChart POST method (bypasses URL length limit)
    payload = {
        "chart": chart_config,
        "backgroundColor": "#1A1A2E",
        "width": 800,
        "height": 400,
        "format": "png"
    }
    
    response = requests.post(
        "https://quickchart.io/chart",
        json=payload,
        timeout=10
    )
    
    if response.status_code != 200:
        raise Exception(f"QuickChart error: {response.status_code}")
    
    return response.content  # Binary image data

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
        logger.info(f"Generating chart for {request.ticker}")
        
        # 1. Generate chart image (binary)
        image_bytes = generate_chart_image(
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
            return {"status": "success", "message": "Chart sent to Telegram"}
        else:
            logger.error(f"Telegram error: {result}")
            return {"status": "error", "message": result.get('description', 'Unknown error')}
            
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "CTOS V1 Chart Engine is running. Use /docs for API documentation."}
