import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY, QDRANT_URL
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# ==============================
# OpenAI Embedding / Qdrant クライアントはモジュール読み込み時に一度だけ作成し、
# 呼び出しのたびに再生成しない（コネクション・クライアントの使い回し）。
# ==============================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

client = QdrantClient(
    url=QDRANT_URL,
)

_vector_stores: dict[str, QdrantVectorStore] = {}


def _get_vector_store(collection_name: str) -> QdrantVectorStore:
    if collection_name not in _vector_stores:
        _vector_stores[collection_name] = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
    return _vector_stores[collection_name]


def vector_search(collection_name, query, k=3):
    # 注意：ここでは検索のみを行い、データの書き込みは行わない。
    # collection は先に build_index.py / get_vector.py を実行して作成しておく必要がある。
    vector_store = _get_vector_store(collection_name)
    docs = vector_store.similarity_search(
        query=query,
        k=k,
    )
    return docs
