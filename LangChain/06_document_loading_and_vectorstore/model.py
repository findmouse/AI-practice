import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL

# ==============================
# 対話モデルの設定（日本語 RAG 用）
# ChatGLM2 は中国語・英語のコーパスが中心で、日本語能力は弱い。
# ここでは GPT-4o に変更：日本語の理解と生成の質が高く、
# build_index.py / chat.py の OpenAIEmbeddings と同じプロバイダーなので、
# 追加のキー申請が不要。
# temperature を低めに設定し、企業向け RAG での事実誤認を減らす。
# ==============================

llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,                          # DeepSeek V3 のモデル名
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,                    # 重要：DeepSeek の API アドレスを指定
    streaming=True                                 # ストリーミング出力に対応
)