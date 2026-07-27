import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL

# 1. DeepSeek API キーを設定
os.environ["DEEPSEEK_API_KEY"] = "your_deepseek_api_key_here"

# 2. Few-Shot Prompt を構築
examples = [
    {"word": "开心", "antonym": "难过"},
    {"word": "高", "antonym": "矮"},
]

example_template = "单词: {word}\n反义词: {antonym}\n"

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="给出每个单词的反义词\n",
    suffix="单词: {input}\n反义词:",
    input_variables=["input"],
    example_separator="\n",
)

prompt_text = few_shot_prompt.format(input="粗")

print("--- 生成された Prompt の内容 ---")
print(prompt_text)
print("*" * 80)

# 3. DeepSeek の大規模言語モデルを初期化
# 注意：DeepSeek のインターフェースは OpenAI 形式と互換性があるため、model_name と openai_api_base を直接指定すればよい
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,                          # 対話モデル
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,                    # DeepSeek の API エンドポイント
    streaming=True
)

# 4. モデルを呼び出して結果を出力
response = llm.invoke(prompt_text)
print("--- モデルの出力結果 ---")
print(response.content)
