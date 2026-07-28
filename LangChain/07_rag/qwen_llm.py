import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TEMPERATURE,
)


def get_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )

if __name__ == '__main__':
    llm = get_llm()

    response = llm.invoke("LangChainを紹介してください。")

    print(response.content)