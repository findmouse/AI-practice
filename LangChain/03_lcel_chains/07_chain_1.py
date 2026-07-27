import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from rich import print
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL
from langchain_core.output_parsers import StrOutputParser

chat = ChatOpenAI(
    model=DEEPSEEK_MODEL,  # DeepSeek V3 のモデル名
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,  # 重要：DeepSeek の API アドレスを指定
    streaming=True  # ストリーミング出力に対応
)

template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
prompt = PromptTemplate(
    input_variables=["lastname"],
    template=template,
)
# prompt_text = prompt.format(lastname="王")

# ここでは統一して | パイプ演算子で構成する LCEL 構文を使用します。
chain = prompt | chat | StrOutputParser()

for chunk in chain.stream({"lastname": "李"}):
    print(chunk, end="", flush=True)

# ==========================================
# シナリオ2：バッチ並行処理 (.batch)
# ==========================================
print("=== 2. バッチ並行処理のデモ ===")

# 複数組の入力データを準備
inputs = [
    {"lastname": "张"},
    {"lastname": "陈"},
    {"lastname": "林"},
]

# .batch() は自動的にマルチスレッド/非同期で並行して API を呼び出し、合計待ち時間を大幅に短縮します
results = chain.batch(inputs)

for inp, res in zip(inputs, results):
    print(f"【姓：{inp['lastname']}】")
    print(res)
    print("-" * 20)
