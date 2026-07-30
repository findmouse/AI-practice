import json
import requests

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "指定された場所の現在の天気を取得する",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "都市名、例：東京、大阪",
                    },
                },
                "required": ["location"],
            },
        }
    },
]


# WMO天気コードを日本語の天気状況に変換する対応表
WEATHER_CODE_MAP = {
    0: "快晴", 1: "晴れ", 2: "一部曇り", 3: "曇り",
    45: "霧", 48: "霧氷",
    51: "弱い霧雨", 53: "霧雨", 55: "強い霧雨",
    56: "弱い着氷性の霧雨", 57: "強い着氷性の霧雨",
    61: "弱い雨", 63: "雨", 65: "強い雨",
    66: "弱い着氷性の雨", 67: "強い着氷性の雨",
    71: "弱い雪", 73: "雪", 75: "強い雪", 77: "霧雪",
    80: "弱いにわか雨", 81: "にわか雨", 82: "激しいにわか雨",
    85: "弱いにわか雪", 86: "強いにわか雪",
    95: "雷雨", 96: "雷雨（軽い雹を伴う）", 99: "雷雨（激しい雹を伴う）",
}


# todo:1.APIを呼び出して天気情報を取得する（日本の都市に対応）
def get_current_weather(location):
    """指定された日本の都市の現在の天気情報を取得する"""
    # Open-Meteoのジオコーディングapiで都市名から緯度経度を取得する
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": location, "count": 1, "language": "ja", "country": "JP"}
    geo_response = requests.get(geo_url, params=geo_params)
    geo_result = geo_response.json()

    weather_info = {}
    results = geo_result.get("results")
    if results:
        place = results[0]
        latitude = place["latitude"]
        longitude = place["longitude"]

        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,weathercode",
            "timezone": "Asia/Tokyo",
        }
        weather_response = requests.get(weather_url, params=weather_params)
        result1 = weather_response.json()
        daily = result1["daily"]
        weather_code = daily["weathercode"][0]
        weather_info = {
            "location": location,
            "date": daily["time"][0],
            "high_temperature": daily["temperature_2m_max"][0],
            "low_temperature": daily["temperature_2m_min"][0],
            "type": WEATHER_CODE_MAP.get(weather_code, "不明"),
        }
    return json.dumps(weather_info, ensure_ascii=False)


# todo: 2.モデルの返答に応じて使用するツール関数を判断する：
"""
# {'content': None, 'role': 'assistant', 'tool_calls':[{'id':'call_8688150014463468290',
'function': {'arguments': '{"location":"東京"}', 'name': 'get_current_weather'}, 'type': 'function'}]}
"""


def parse_response(response):
    response_message = response.choices[0].message
    # 関数呼び出しが必要かどうかを判定する
    if response_message.tool_calls:
        # 関数を呼び出す
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # この例では関数は一つだけだが、複数の関数を登録することも可能
        function_name = response_message.tool_calls[0].function.name
        # 関数を取得する
        fuction_to_call = available_functions[function_name]
        # 関数の引数を取得する
        function_args = json.loads(response_message.tool_calls[0].function.arguments)

        function_response = fuction_to_call(
            location=function_args.get("location"),
        )
        return function_response
