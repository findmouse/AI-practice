import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flashrank import Ranker, RerankRequest
from langchain_core.tools import tool

from vector_search import vector_search

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COLLECTION_NAME = "pku"
RETRIEVE_K = 5      # 先粗召回 5 条
RERANK_TOP_N = 3    # 重排序后取前 3 条喂给 LLM

# リランカーはモジュール読み込み時に一度だけロードし、呼び出しのたびに再生成しない。
_reranker = Ranker(cache_dir=os.path.join(BASE_DIR, ".flashrank_cache"))


@tool
def search_knowledge_base(query: str) -> str:
    """北京大学（PKU）に関する事実を社内知識ベースから検索する。

    創立年、学部構成、キャンパスの歴史など、具体的な事実確認が必要な質問に
    答える前に呼び出すこと。世間話や、すでに会話履歴に出てきた内容についての
    質問には使わなくてよい。
    """
    docs = vector_search(COLLECTION_NAME, query, k=RETRIEVE_K)
    if not docs:
        return "関連する資料が見つかりませんでした。"

    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    reranked = _reranker.rerank(RerankRequest(query=query, passages=passages))

    top_passages = reranked[:RERANK_TOP_N]
    return "\n\n".join(p["text"] for p in top_passages)
