# Weather Agent

A CAL agent that provides weather information for any location worldwide using the free [Open-Meteo](https://open-meteo.com/) API.

## Features

- **Location Search**: Find any city by name with country/state disambiguation
- **Current Weather**: Real-time conditions including temperature, humidity, wind, and precipitation
- **Daily Forecast**: Up to 16-day forecasts with highs, lows, and precipitation chances
- **Hourly Forecast**: Detailed 48-hour forecasts for precise planning
- **Multi-Location Comparison**: Compare weather across multiple cities side-by-side

## Setup

1. Create a `.env` file with your Gemini API key:

```bash
GEMINI_API_KEY=your_gemini_api_key
```

> **Note**: No weather API key is required! Open-Meteo is free and open-source.

2. Install dependencies:

```bash
pip install CAL python-dotenv httpx
```

3. Run the agent:

```bash
python agent.py
```

## Example Usage

```
You: What's the weather in Paris?

# Agent searches for Paris coordinates, then fetches current weather

You: Will it rain tomorrow?

# Agent fetches daily forecast and interprets precipitation chances

You: Compare New York and Miami weather

# Agent looks up both cities and shows side-by-side comparison

You: Hourly forecast for the next 12 hours

# Agent shows detailed hour-by-hour breakdown
```

## File Structure

```
weather_agent/
├── agent.py       # Main agent setup and interactive loop
├── tools.py       # Weather API tools (Open-Meteo integration)
├── prompt.py      # System prompts and guidelines
└── README.md      # This file
```

## Tools Overview

| Tool | Purpose |
|------|---------|
| `search_location` | Convert city name to coordinates |
| `get_current_weather` | Current conditions (temp, wind, humidity) |
| `get_weather_forecast` | Daily forecast (1-16 days) |
| `get_hourly_forecast` | Hourly forecast (1-48 hours) |
| `compare_weather` | Compare multiple locations |

## Key CAL Concepts Demonstrated

### 1. External API Integration

```python
@tool
async def get_current_weather(latitude: float, longitude: float, location_name: str = ""):
    """Get current weather conditions."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": latitude, "longitude": longitude, ...}
        )
        data = response.json()
    
    return {
        "content": [{"type": "text", "text": formatted_report}],
        "metadata": {"temperature": data["current"]["temperature_2m"]}
    }
```

### 2. Tool Chaining

The agent learns to chain tools effectively:
1. First call `search_location("Paris")` to get coordinates
2. Then call `get_current_weather(48.85, 2.35, "Paris")` with those coordinates

### 3. Contextual Responses

The system prompt teaches the agent to interpret weather data contextually:
- "It's 5°C" → "It's cold, dress warmly"
- "80% precipitation" → "Bring an umbrella"

## Open-Meteo API

This example uses [Open-Meteo](https://open-meteo.com/), a free weather API that:
- Requires no API key or registration
- Provides global coverage
- Offers hourly and daily forecasts
- Includes historical weather data
- Has generous rate limits for personal use

## Customization Ideas

- Add severe weather alerts
- Include air quality data (Open-Meteo provides this)
- Add sunrise/sunset information
- Store favorite locations
- Historical weather comparison
