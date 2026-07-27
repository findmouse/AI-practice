import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL
from rich import print

# 1. DeepSeek の大規模言語モデルをインスタンス化
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,  #deepseek-v4-pro
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,  # 重要：DeepSeek の API アドレスを指定
    streaming=True  # ストリーミング出力に対応
)

# 2. Agent が使用できるツールのリストを設定
# DuckDuckGo 検索ツール（API Key 不要）
tools = [
    DuckDuckGoSearchRun(name="Search"),
]

# 3. 標準的な ReAct プロンプトテンプレート（hub.pull("hwchase17/react") と同等）
# この Prompt は思考のフォーマットを規定しています：Thought -> Action -> Action Input -> Observation -> Final Answer
# テンプレートを直接組み込み、実行のたびに LangChain Hub にリクエストするのを回避（新バージョンの langsmith はデフォルトで公開 Prompt への明示的な信頼が必要）
prompt = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

# 4. ReAct Agent と AgentExecutor 実行器を構築
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # verbose を有効にするとモデルの思考チェーン（Thought/Action/Observation）をリアルタイムで確認できる
    handle_parsing_errors=True,  # 解析失敗時を自動的に処理
)

# 5. Agent タスクを実行
query = "中国目前有多少人口"
response = agent_executor.invoke({"input": query})

print("\n=== 最終回答 ===")
print(response["output"])
