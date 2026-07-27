import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL

# 1. 元のテンプレートを作成
template_str = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
注意: 文字不要超过50个字符"""

# 3. 元のテンプレートから LangChain のプロンプトテンプレートを作成
promp_emplate = ChatPromptTemplate.from_template(template_str)

# 注意：パラメータを渡す際、flower_name はリスト ["玫瑰"] ではなく文字列 "玫瑰" にすることを推奨
prompt = promp_emplate.format_messages(flower_name="玫瑰", price='50')
print('prompt-->', prompt)

# 4. DeepSeek モデルをインスタンス化（ChatOpenAI コンポーネントを使用）
chat = ChatOpenAI(
    model=DEEPSEEK_MODEL,                          # 対話モデル
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,                    # DeepSeek の API エンドポイント
    streaming=True
)

# 5. モデルを呼び出して結果を出力（新しいバージョンの LangChain では invoke の使用を推奨）
result = chat.invoke(prompt)
print(result)
print("\n生成された内容：", result.content)
