import httpx
from CAL import tool
from datetime import datetime, timedelta


@tool
async def search_location(city_name: str):
    """Search for a location to get its coordinates for weather queries.
    
    Args:
        city_name: Name of the city or location to search for
    
    Returns:
        List of matching locations with coordinates and country info
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city_name,
                "count": 5,
                "language": "en",
                "format": "json"
            }
        )
        data = response.json()
    
    results = data.get("results", [])
    
    if not results:
        return {
            "content": [{"type": "text", "text": f"No locations found for '{city_name}'"}],
            "metadata": {"found": False}
        }
    
    formatted = []
    for loc in results:
        formatted.append(
            f"**{loc['name']}**, {loc.get('admin1', '')}, {loc.get('country', '')}\n"
            f"  Coordinates: {loc['latitude']}, {loc['longitude']}\n"
            f"  Timezone: {loc.get('timezone', 'N/A')}"
        )
    
    return {
        "content": [{"type": "text", "text": "\n\n".join(formatted)}],
        "metadata": {
            "found": True,
            "count": len(results),
            "locations": [
                {"name": r["name"], "lat": r["latitude"], "lon": r["longitude"]}
                for r in results
            ]
        }
    }


@tool
async def get_current_weather(latitude: float, longitude: float, location_name: str = ""):
    """Get current weather conditions for a specific location.
    
    Args:
        latitude: Latitude coordinate of the location
        longitude: Longitude coordinate of the location
        location_name: Optional name of the location for display
    
    Returns:
        Current weather conditions including temperature, wind, and conditions
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
                "timezone": "auto"
            }
        )
        data = response.json()
    
    current = data.get("current", {})
    units = data.get("current_units", {})
    
    weather_codes = {
        0: "Clear sky ☀️",
        1: "Mainly clear 🌤️",
        2: "Partly cloudy ⛅",
        3: "Overcast ☁️",
        45: "Foggy 🌫️",
        48: "Depositing rime fog 🌫️",
        51: "Light drizzle 🌧️",
        53: "Moderate drizzle 🌧️",
        55: "Dense drizzle 🌧️",
        61: "Slight rain 🌧️",
        63: "Moderate rain 🌧️",
        65: "Heavy rain 🌧️",
        71: "Slight snow 🌨️",
        73: "Moderate snow 🌨️",
        75: "Heavy snow 🌨️",
        80: "Slight rain showers 🌦️",
        81: "Moderate rain showers 🌦️",
        82: "Violent rain showers 🌦️",
        95: "Thunderstorm ⛈️",
        96: "Thunderstorm with slight hail ⛈️",
        99: "Thunderstorm with heavy hail ⛈️",
    }
    
    weather_desc = weather_codes.get(current.get("weather_code", 0), "Unknown")
    location_display = f" for {location_name}" if location_name else ""
    
    report = f"""## Current Weather{location_display}

**Conditions**: {weather_desc}
**Temperature**: {current.get('temperature_2m', 'N/A')}{units.get('temperature_2m', '°C')}
**Feels Like**: {current.get('apparent_temperature', 'N/A')}{units.get('apparent_temperature', '°C')}
**Humidity**: {current.get('relative_humidity_2m', 'N/A')}{units.get('relative_humidity_2m', '%')}
**Wind**: {current.get('wind_speed_10m', 'N/A')} {units.get('wind_speed_10m', 'km/h')} from {current.get('wind_direction_10m', 'N/A')}°
**Wind Gusts**: {current.get('wind_gusts_10m', 'N/A')} {units.get('wind_gusts_10m', 'km/h')}
**Precipitation**: {current.get('precipitation', 'N/A')} {units.get('precipitation', 'mm')}

*Updated: {current.get('time', 'N/A')}*
"""
    
    return {
        "content": [{"type": "text", "text": report}],
        "metadata": {
            "temperature": current.get("temperature_2m"),
            "weather_code": current.get("weather_code"),
            "location": location_name
        }
    }


@tool
async def get_weather_forecast(latitude: float, longitude: float, days: int = 7, location_name: str = ""):
    """Get weather forecast for the next several days.
    
    Args:
        latitude: Latitude coordinate of the location
        longitude: Longitude coordinate of the location
        days: Number of days to forecast (1-16, default 7)
        location_name: Optional name of the location for display
    
    Returns:
        Daily weather forecast with highs, lows, and conditions
    """
    days = max(1, min(16, days))
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,sunrise,sunset",
                "timezone": "auto",
                "forecast_days": days
            }
        )
        data = response.json()
    
    daily = data.get("daily", {})
    units = data.get("daily_units", {})
    
    weather_codes = {
        0: "Clear ☀️", 1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
        45: "Foggy 🌫️", 48: "Rime fog 🌫️", 51: "Light drizzle 🌧️", 53: "Drizzle 🌧️",
        55: "Dense drizzle 🌧️", 61: "Light rain 🌧️", 63: "Rain 🌧️", 65: "Heavy rain 🌧️",
        71: "Light snow 🌨️", 73: "Snow 🌨️", 75: "Heavy snow 🌨️", 80: "Rain showers 🌦️",
        81: "Moderate showers 🌦️", 82: "Heavy showers 🌦️", 95: "Thunderstorm ⛈️",
        96: "Thunderstorm + hail ⛈️", 99: "Severe thunderstorm ⛈️",
    }
    
    location_display = f" for {location_name}" if location_name else ""
    lines = [f"## {days}-Day Forecast{location_display}\n"]
    
    dates = daily.get("time", [])
    for i, date in enumerate(dates):
        code = daily.get("weather_code", [])[i] if i < len(daily.get("weather_code", [])) else 0
        weather = weather_codes.get(code, "Unknown")
        
        high = daily.get("temperature_2m_max", [])[i] if i < len(daily.get("temperature_2m_max", [])) else "N/A"
        low = daily.get("temperature_2m_min", [])[i] if i < len(daily.get("temperature_2m_min", [])) else "N/A"
        precip = daily.get("precipitation_sum", [])[i] if i < len(daily.get("precipitation_sum", [])) else 0
        precip_prob = daily.get("precipitation_probability_max", [])[i] if i < len(daily.get("precipitation_probability_max", [])) else 0
        wind = daily.get("wind_speed_10m_max", [])[i] if i < len(daily.get("wind_speed_10m_max", [])) else "N/A"
        
        day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%A, %b %d")
        
        lines.append(f"### {day_name}")
        lines.append(f"**{weather}**")
        lines.append(f"High: {high}°C / Low: {low}°C")
        lines.append(f"Precipitation: {precip}mm ({precip_prob}% chance)")
        lines.append(f"Wind: up to {wind} km/h")
        lines.append("")
    
    return {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "metadata": {
            "days": days,
            "location": location_name,
            "dates": dates
        }
    }


@tool
async def get_hourly_forecast(latitude: float, longitude: float, hours: int = 24, location_name: str = ""):
    """Get detailed hourly weather forecast.
    
    Args:
        latitude: Latitude coordinate of the location
        longitude: Longitude coordinate of the location
        hours: Number of hours to forecast (1-48, default 24)
        location_name: Optional name of the location for display
    
    Returns:
        Hourly forecast with temperature, precipitation, and conditions
    """
    hours = max(1, min(48, hours))
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m",
                "timezone": "auto",
                "forecast_hours": hours
            }
        )
        data = response.json()
    
    hourly = data.get("hourly", {})
    
    weather_codes = {
        0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️", 45: "🌫️", 48: "🌫️",
        51: "🌧️", 53: "🌧️", 55: "🌧️", 61: "🌧️", 63: "🌧️", 65: "🌧️",
        71: "🌨️", 73: "🌨️", 75: "🌨️", 80: "🌦️", 81: "🌦️", 82: "🌦️",
        95: "⛈️", 96: "⛈️", 99: "⛈️",
    }
    
    location_display = f" for {location_name}" if location_name else ""
    lines = [f"## Hourly Forecast{location_display} (next {hours} hours)\n"]
    lines.append("| Time | Temp | Weather | Precip % | Wind |")
    lines.append("|------|------|---------|----------|------|")
    
    times = hourly.get("time", [])[:hours]
    for i, time_str in enumerate(times):
        temp = hourly.get("temperature_2m", [])[i] if i < len(hourly.get("temperature_2m", [])) else "N/A"
        code = hourly.get("weather_code", [])[i] if i < len(hourly.get("weather_code", [])) else 0
        precip_prob = hourly.get("precipitation_probability", [])[i] if i < len(hourly.get("precipitation_probability", [])) else 0
        wind = hourly.get("wind_speed_10m", [])[i] if i < len(hourly.get("wind_speed_10m", [])) else "N/A"
        
        weather_icon = weather_codes.get(code, "❓")
        time_display = datetime.fromisoformat(time_str).strftime("%a %H:%M")
        
        lines.append(f"| {time_display} | {temp}°C | {weather_icon} | {precip_prob}% | {wind} km/h |")
    
    return {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "metadata": {
            "hours": hours,
            "location": location_name
        }
    }


@tool
async def compare_weather(locations: str):
    """Compare current weather across multiple locations.
    
    Args:
        locations: Comma-separated list of "city_name:lat:lon" entries 
                   (e.g., "New York:40.71:-74.01,London:51.51:-0.13")
    
    Returns:
        Side-by-side comparison of current conditions
    """
    location_list = []
    for loc in locations.split(","):
        parts = loc.strip().split(":")
        if len(parts) >= 3:
            location_list.append({
                "name": parts[0].strip(),
                "lat": float(parts[1]),
                "lon": float(parts[2])
            })
    
    if not location_list:
        return {
            "content": [{"type": "text", "text": "No valid locations provided. Format: 'City:lat:lon,City2:lat2:lon2'"}],
            "metadata": {"error": True}
        }
    
    weather_codes = {
        0: "Clear ☀️", 1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
        45: "Foggy 🌫️", 51: "Drizzle 🌧️", 61: "Rain 🌧️", 71: "Snow 🌨️", 95: "Storm ⛈️",
    }
    
    results = []
    async with httpx.AsyncClient() as client:
        for loc in location_list:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": loc["lat"],
                    "longitude": loc["lon"],
                    "current": "temperature_2m,weather_code,wind_speed_10m",
                    "timezone": "auto"
                }
            )
            data = response.json()
            current = data.get("current", {})
            
            code = current.get("weather_code", 0)
            weather = weather_codes.get(code, weather_codes.get(code // 10 * 10, "Unknown"))
            
            results.append({
                "name": loc["name"],
                "temp": current.get("temperature_2m", "N/A"),
                "weather": weather,
                "wind": current.get("wind_speed_10m", "N/A")
            })
    
    lines = ["## Weather Comparison\n"]
    lines.append("| Location | Temperature | Conditions | Wind |")
    lines.append("|----------|-------------|------------|------|")
    
    for r in results:
        lines.append(f"| {r['name']} | {r['temp']}°C | {r['weather']} | {r['wind']} km/h |")
    
    return {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "metadata": {"locations_compared": len(results)}
    }
