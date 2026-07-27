import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL

llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,  # deepseek-v4-pro
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,  # 重要：DeepSeek の API アドレスを指定
    streaming=True  # ストリーミング出力に対応
)
