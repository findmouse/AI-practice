import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL
from rich import print

# 1. DeepSeek の大規模言語モデルをインスタンス化
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,  # DeepSeek V3 のモデル名
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,  # 重要：DeepSeek の API アドレスを指定
    streaming=True  # ストリーミング出力に対応
)

# 2. 履歴メッセージのプレースホルダーを持つ ChatPromptTemplate を定義
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助人为乐的AI助手。"),
    MessagesPlaceholder(variable_name="history"),  # 過去の対話履歴のプレースホルダー
    ("human", "{input}"),
])

# 基础链
base_chain = prompt | llm | StrOutputParser()

# 3. インメモリデータベース：異なる session_id の対話履歴を保存するために使用
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 4. メモリ機能付きのチェーンにラップ
# 基本チェーン、チャット履歴チェーン、そして入力と履歴を識別するキー名を含む
conversation = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 5. 現在のセッションの session_id を設定して対話を実行
config = {"configurable": {"session_id": "user_123"}}

result1 = conversation.invoke({"input": "小明有1只猫"}, config=config)
print(result1)
print("*" * 80)

result2 = conversation.invoke({"input": "小刚有2只狗"}, config=config)
print(result2)
print("*" * 80)

result3 = conversation.invoke({"input": "小明和小刚相比，谁的宠物多?"}, config=config)
print(result3)
print("*" * 80)
