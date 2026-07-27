import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL

# クライアントを初期化。base_url は必ず DeepSeek を指すこと
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE
)

response = client.chat.completions.create(
    model=DEEPSEEK_MODEL,  # 対話モデル（推論モデルが必要な場合は deepseek-reasoner を使用可）
    messages=[
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "再帮帮我讲个笑话吧，换一个吧。不要关于数学书的。"}
    ],
    stream=False
)

print(response.choices[0].message.content)
