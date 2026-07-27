import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL

# 1. 大規模言語モデルを定義
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,  # deepseek-v4-pro
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,  # 重要：DeepSeek の API アドレスを指定
    streaming=True  # ストリーミング出力に対応
)

# 2. 2つの Prompt テンプレートを定義
first_prompt = PromptTemplate(
    input_variables=["lastname"],
    template="我的邻居姓{lastname}，他生了个儿子，请只给出他儿子的大名（不要包含其他多余字句）。",
)

second_prompt = PromptTemplate(
    input_variables=["child_name"],
    template="邻居的儿子名字叫{child_name}，请给他起一个小名，并说明理由。",
)

# 3. LCEL のパイプ演算子で2つのチェーンを連結
# データの流れ：
# first_prompt -> llm -> StrOutputParser() で正式な名前のテキストを抽出
# -> 辞書 {"child_name": ...} にラップ -> second_prompt -> llm -> StrOutputParser() で最終結果を取得
# overall_chain = (
#         first_prompt
#         | llm
#         | StrOutputParser()
#         | (lambda child_name: {"child_name": child_name.strip()})
#         | second_prompt
#         | llm
#         | StrOutputParser()
# )

overall_chain = (
    {"child_name": first_prompt | llm | StrOutputParser()}
    | second_prompt
    | llm
    | StrOutputParser()
)

# 4. チェーンを実行
result = overall_chain.invoke({"lastname": "刘"})
print(result)
