import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from rich import print
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL



chat = ChatOpenAI(
    model=DEEPSEEK_MODEL,                          # DeepSeek V3 のモデル名
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,                    # 重要：DeepSeek の API アドレスを指定
    streaming=True                                 # ストリーミング出力に対応
)

template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
prompt = PromptTemplate.from_template(template)
prompt_text = prompt.format(lastname="王")

result = chat.invoke(prompt_text)
print(result)
