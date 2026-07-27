import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ==============================
# OpenAI Embedding の設定
# text-embedding-3-small    1536
# text-embedding-3-large    3072
# ==============================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

# ==============================
# Qdrant に接続
# 注意：ここでは検索のみを行い、データの書き込みは行わない。
# collection は先に build_index.py を実行して作成しておく必要がある。
# ==============================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pku"

client = QdrantClient(
    url=QDRANT_URL,
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# ==============================
# 検索
# ==============================
query = "1937年北京大学发生了什么？"

docs = vector_store.similarity_search(
    query=query,
    k=3,
)

print("=" * 50)

for i, doc in enumerate(docs, start=1):
    print(f"結果{i}")
    print(doc.page_content)
    print("-" * 50)
