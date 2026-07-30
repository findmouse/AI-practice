import os
import sys
from pkgutil import resolve_name

from sympy import rsolve

from LangChain.config import DEEPSEEK_MODEL

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout.reconfigure(encoding='utf-8')


from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL
from openai import OpenAI
from sql_function_tools import *
from rich import print

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)

def chat_completion_request(messages, tools=None, tool_choice = None,model = DEEPSEEK_MODEL):
    try:
        respnse =client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return respnse
    except Exception as e:
        return e


def main():
    # メッセージを作成
    messages = []
    messages.append(
        {
            'role':'system',
            'content':'業務データベースに対してSQLクエリを生成することでユーザーの質問に回答してください'
         }
    )
    messages.append(
        {
            'role': 'user',
            'content': '最も給料の高い社員の名前とその給料を教えてください'
        }
    )
    response = chat_completion_request(messages,tools= tools, tool_choice='auto')
    print(f"response-->{response}")
    assistant_message = response.choices[0].message
    messages.append(assistant_message.model_dump())
    function_name = assistant_message.tool_calls[0].function.name
    function_id = assistant_message.tool_calls[0].id
    function_response = parse_response(response)
    messages.append(
        {
            'role':'tool',
            'tool_call_id':function_id,
            'name':function_name,
            'content':str(function_response),
        }
    )
    last_response = chat_completion_request(messages,tools= tools, tool_choice='auto')
    print(f"last_response-->{last_response}")



if __name__ == '__main__':
    main()
