import json
import os
import requests

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_plane_number",
            "description": "出発地、目的地、日付に基づいて、該当する日付の便名を検索する",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "description": "出発地",
                        "type": "string"
                    },
                    "end": {
                        "description": "目的地",
                        "type": "string"
                    },
                    "date": {
                        "description": "日付",
                        "type": "string",
                    }
                },
                "required": ["start", "end", "date"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket_price",
            "description": "特定の便のある日の価格を検索する",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "description": "便名",
                        "type": "string"
                    },
                    "date": {
                        "description": "日付",
                        "type": "string",
                    }
                },
                "required": [ "number", "date"]
            },
        }
    },
]


def get_plane_number(date, start, end):
    plane_number = {
        "北京": {
            "深圳": "126",
            "广州": "356",
        },
        "郑州": {
            "北京": "1123",
            "天津": "3661",
        }
    }
    return {"date": date, "number": plane_number[start][end]}


def get_ticket_price(date: str, number: str):
    print(date)
    print(number)
    return {"ticket_price": "668"}


def call_function(tool_call):
    '''

    :param tool_call: モデルの返答結果に含まれる一つの tool_call
    :return: 該当する tool_call に対応する関数の実行結果を返す
    '''
    available_functions = {
        "get_plane_number": get_plane_number,
        "get_ticket_price": get_ticket_price,
    }
    function_name = tool_call.function.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)
    return function_to_call(**function_args)
