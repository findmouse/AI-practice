import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout.reconfigure(encoding='utf-8')
from openai import OpenAI
from airplane_function_tools import *
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
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def main():
    messages = []
    messages.append({"role": "system",
                     "content": "あなたは航空券検索アシスタントです。ユーザーの質問に基づいて回答してください。ただし、関数に渡すパラメータの値を勝手に推測したり作成したりしないでください。ユーザーの説明が不明確な場合は、必要な情報を提供するようユーザーに求めてください"})
    messages.append({"role": "user", "content": "2024年4月2日、鄭州から北京までの航空券の価格を調べてください"})

    # モデルがツール関数の呼び出しを必要としなくなるまでループで呼び出す
    while True:
        response = chat_completion_request(messages, tools=tools, tool_choice="auto")
        assistant_message = response.choices[0].message
        print(f'assistant_message-->{assistant_message}')
        messages.append(assistant_message.model_dump())

        if not assistant_message.tool_calls:
            break

        for tool_call in assistant_message.tool_calls:
            function_response = call_function(tool_call)
            print(f'function_response--》{function_response}')
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(function_response, ensure_ascii=False),
                }
            )

    print(f'last_response--》{messages[-1]}')


if __name__ == '__main__':
    main()
