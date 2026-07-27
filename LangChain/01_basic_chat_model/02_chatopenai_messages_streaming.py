import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL

# ChatOpenAI インスタンスを初期化し、エンドポイントとモデルを DeepSeek に向ける
chat = ChatOpenAI(
    model=DEEPSEEK_MODEL,                          # DeepSeek V3 のモデル名
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,                    # 重要：DeepSeek の API アドレスを指定
    streaming=True                                 # ストリーミング出力に対応
)

# 3. メッセージを構築（チュートリアル画像の使い方と完全に一致）
# messages = [
#     HumanMessage(content="唐詩を一首書いてください")
# ]

# システムプロンプト（SystemMessage）を追加する場合は、次のように書けます：
messages = [
    SystemMessage(content="你是一位精通古典诗词的文学大师"),
    HumanMessage(content="给我写一首唐诗")
]

# 4. モデルを呼び出して結果を出力
res = chat.invoke(messages)  # 注意：新しいバージョンの LangChain では chat() を直接呼ぶ代わりに .invoke() の使用が推奨されています
print(res.content)
