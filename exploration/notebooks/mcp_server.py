from mcp.server.fastmcp import FastMCP
import httpx
mcp = FastMCP("my-first-mcp-server")


# Tool 1: simple arithmetic
@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """
    Adds two numbers and returns the result.
    Use this when the user asks to add or sum two values.
    """
    return a + b


# Tool 2: string operation
@mcp.tool()
def reverse_string(text: str) -> str:
    """
    Reverses the characters in a string.
    Use this when the user wants to reverse text.
    """
    return text[::-1]


# Tool 3: multiple inputs, structured logic
@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """
    Calculates BMI given weight in kilograms and height in meters.
    Returns the BMI value and the category (Underweight, Normal, Overweight, Obese).
    """
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return {"bmi": round(bmi, 2), "category": category}


# --- Function tools from Part 3 ---
# --- API tool ---

@mcp.tool()
async def get_current_weather(latitude: float, longitude: float) -> dict:
    """
    Fetches current weather for the given latitude and longitude.
    Returns temperature in Celsius, wind speed in km/h, and weather condition code.
    Use this when the user asks about current weather at a location.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m", "wind_speed_10m", "weather_code"],
        "forecast_days": 1,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()

    current = data["current"]
    return {
        "temperature_celsius": current["temperature_2m"],
        "wind_speed_kmh": current["wind_speed_10m"],
        "weather_code": current["weather_code"],
        "latitude": latitude,
        "longitude": longitude,
    }

if __name__ == "__main__":
    print("Starting MCP server...")
    mcp.run(transport="stdio")