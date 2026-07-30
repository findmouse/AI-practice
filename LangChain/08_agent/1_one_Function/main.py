import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout.reconfigure(encoding='utf-8')
from openai import OpenAI
from tools import *
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL
from rich import print

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)


def chat_completion_request(messages, tools=None, tool_choice=None, model=DEEPSEEK_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("ChatCompletionレスポンスの生成に失敗しました")
        print(f"Exception: {e}")
        return e


def main(position):
    messages = []
    messages.append({"role": "system",
                     "content": "あなたは天気予報アシスタントです。ユーザーが指定した場所に基づいて現地の天気状況を回答してください。ユーザーの質問内容が不明確な場合は、勝手に内容を作らず、ユーザーに明確な入力を促してください"})
    messages.append({"role": "user",
                     "content": f"今日の{position}の天気はどうですか"})
    response = chat_completion_request(
        messages, tools=tools, tool_choice="auto"
    )
    assistant_message = response.choices[0].message
    # アシスタントの返答を会話履歴に追加する
    messages.append(assistant_message.model_dump())
    function_name = response.choices[0].message.tool_calls[0].function.name
    function_id = response.choices[0].message.tool_calls[0].id
    function_response = parse_response(response)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": function_id,
            "name": function_name,
            "content": function_response,
        }

    )  # 関数の実行結果を会話履歴に追加する
    last_response = chat_completion_request(
        messages, tools=tools, tool_choice="auto"
    )
    print(f'last_response--》{last_response.choices[0].message}')


if __name__ == '__main__':
    while True:
        city = input("天気を調べたい日本の都市名を入力してください（終了する場合はexit）：")
        if city == 'exit' or city == 'quit':
            break

        main(city)
