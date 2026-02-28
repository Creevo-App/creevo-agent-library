"""Custom tools for the Weather Agent using Open-Meteo API."""

import sys
from pathlib import Path

# Add src directory to path to use local CAL code
_src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import httpx

from CAL import tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32


def get_weather_description(code: int) -> str:
    """Get human-readable weather description from WMO code."""
    return WEATHER_CODE_DESCRIPTIONS.get(code, f"Unknown condition (code {code})")


def get_wind_direction(degrees: int) -> str:
    """Convert wind direction degrees to cardinal direction."""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(degrees / 22.5) % 16
    return directions[index]


@tool
async def geocode_city(city_name: str, country: str = "") -> dict:
    """
    Look up geographic coordinates for a city.
    
    Args:
        city_name: The name of the city to look up
        country: Optional country name or code to narrow results
    
    Returns:
        Location information including coordinates for matching cities
    """
    params = {
        "name": city_name,
        "count": 5,
        "language": "en",
        "format": "json",
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(GEOCODING_URL, params=params)
        response.raise_for_status()
        data = response.json()
    
    if "results" not in data or not data["results"]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"No locations found for '{city_name}'. Please check the spelling or try a different city name.",
                }
            ],
            "metadata": {"error": True, "query": city_name},
        }
    
    results = data["results"]
    
    if country:
        country_lower = country.lower()
        filtered = [
            r for r in results
            if country_lower in r.get("country", "").lower()
            or country_lower in r.get("country_code", "").lower()
        ]
        if filtered:
            results = filtered
    
    locations = []
    for r in results[:5]:
        loc_parts = [r.get("name", "Unknown")]
        if r.get("admin1"):
            loc_parts.append(r["admin1"])
        if r.get("country"):
            loc_parts.append(r["country"])
        
        locations.append({
            "name": ", ".join(loc_parts),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "country": r.get("country"),
            "country_code": r.get("country_code"),
            "timezone": r.get("timezone"),
        })
    
    location_text = "Found the following location(s):\n\n"
    for i, loc in enumerate(locations, 1):
        location_text += f"{i}. **{loc['name']}**\n"
        location_text += f"   Coordinates: {loc['latitude']:.4f}°N, {loc['longitude']:.4f}°E\n"
        location_text += f"   Timezone: {loc['timezone']}\n\n"
    
    return {
        "content": [{"type": "text", "text": location_text}],
        "metadata": {"locations": locations, "query": city_name},
    }


@tool
async def get_current_weather(latitude: float, longitude: float, location_name: str = "") -> dict:
    """
    Get current weather conditions for a location.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        location_name: Optional name of the location for display
    
    Returns:
        Current weather conditions including temperature, humidity, wind, etc.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "is_day",
            "precipitation",
            "rain",
            "showers",
            "snowfall",
            "weather_code",
            "cloud_cover",
            "pressure_msl",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
        ],
        "timezone": "auto",
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(WEATHER_URL, params=params)
        response.raise_for_status()
        data = response.json()
    
    if "current" not in data:
        return {
            "content": [
                {
                    "type": "text",
                    "text": "Unable to retrieve weather data for this location.",
                }
            ],
            "metadata": {"error": True},
        }
    
    current = data["current"]
    units = data.get("current_units", {})
    
    temp_c = current.get("temperature_2m", 0)
    temp_f = celsius_to_fahrenheit(temp_c)
    feels_like_c = current.get("apparent_temperature", 0)
    feels_like_f = celsius_to_fahrenheit(feels_like_c)
    
    weather_code = current.get("weather_code", 0)
    conditions = get_weather_description(weather_code)
    
    wind_speed = current.get("wind_speed_10m", 0)
    wind_dir = get_wind_direction(current.get("wind_direction_10m", 0))
    wind_gusts = current.get("wind_gusts_10m", 0)
    
    humidity = current.get("relative_humidity_2m", 0)
    cloud_cover = current.get("cloud_cover", 0)
    pressure = current.get("pressure_msl", 0)
    
    precipitation = current.get("precipitation", 0)
    rain = current.get("rain", 0)
    snow = current.get("snowfall", 0)
    
    is_day = current.get("is_day", 1)
    time_of_day = "day" if is_day else "night"
    
    location_str = location_name if location_name else f"{latitude:.2f}°N, {longitude:.2f}°E"
    
    weather_text = f"""## Current Weather for {location_str}

**Conditions:** {conditions}
**Time:** {time_of_day.capitalize()}time

### Temperature
- **Current:** {temp_c:.1f}°C ({temp_f:.1f}°F)
- **Feels Like:** {feels_like_c:.1f}°C ({feels_like_f:.1f}°F)

### Atmosphere
- **Humidity:** {humidity}%
- **Cloud Cover:** {cloud_cover}%
- **Pressure:** {pressure:.1f} hPa

### Wind
- **Speed:** {wind_speed} km/h from {wind_dir}
- **Gusts:** {wind_gusts} km/h
"""
    
    if precipitation > 0 or rain > 0 or snow > 0:
        weather_text += f"""
### Precipitation
- **Total:** {precipitation} mm
"""
        if rain > 0:
            weather_text += f"- **Rain:** {rain} mm\n"
        if snow > 0:
            weather_text += f"- **Snow:** {snow} cm\n"
    
    return {
        "content": [{"type": "text", "text": weather_text}],
        "metadata": {
            "location": location_str,
            "temperature_c": temp_c,
            "temperature_f": temp_f,
            "conditions": conditions,
            "weather_code": weather_code,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_direction": wind_dir,
        },
    }
