import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_openai import ChatOpenAI
from langchain_community.llms import QianfanLLMEndpoint
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL

# API Key と DeepSeek のドメインを設定
chat = ChatOpenAI(
    model=DEEPSEEK_MODEL,                          # 対話モデル
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,                    # DeepSeek の API エンドポイント
    streaming=True
)

res = chat.invoke("帮我讲个笑话吧")
print(res.content)
