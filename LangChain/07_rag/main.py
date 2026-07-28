import argparse
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE
from rag_tool import search_knowledge_base

SYSTEM_PROMPT = """あなたは北京大学（PKU）について答えるアシスタントです。

- 事実確認が必要な質問には、必ず search_knowledge_base ツールで知識ベースを
  検索してから回答してください。
- 検索しても分からない場合は、正直に「分かりません」と答え、推測で答えを
  作らないでください。
- すでに会話に出てきた内容についての質問や世間話には、ツールを使わず
  会話履歴だけで答えて構いません。
"""

# LLM とエージェントはプロセス起動時に一度だけ構築し、以降のリクエストで使い回す。
# checkpointer が thread_id ごとに会話履歴（Memory）を保持するので、
# 同じ thread_id で invoke を繰り返せば複数ターンの会話が成立する。
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=TEMPERATURE,
)

agent = create_agent(
    model=llm,
    tools=[search_knowledge_base],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)


def ask(question: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    return result["messages"][-1].content


def chat_loop() -> None:
    thread_id = str(uuid.uuid4())
    print("複数ターンの会話ができます。'exit' または 'quit' で終了します。")
    while True:
        try:
            question = input("あなた: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        answer = ask(question, thread_id)
        print(f"AI: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RAG 問答（複数ターン対応）")
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="単発質問。指定しない場合は対話モードで起動する",
    )
    args = parser.parse_args()

    if args.question:
        print("=== 問題 ===")
        print(args.question)
        print("=== 回答 ===")
        print(ask(args.question, thread_id=str(uuid.uuid4())))
    else:
        chat_loop()
