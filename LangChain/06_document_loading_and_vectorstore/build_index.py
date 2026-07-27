import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY
from langchain_text_splitters import CharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

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
# ==============================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pku"

client = QdrantClient(
    url=QDRANT_URL,
)

# ==============================
# VectorStore を作成
# Collection が存在しない場合は先に作成する
# Distance.Cosine  Distance.Euclidean  Distance.Dot
# ==============================
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# ==============================
# テキストを読み込む
# ==============================
with open("./data/pku.txt", encoding="utf-8") as f:
    text = f.read()

# ==============================
# テキスト分割
# chunk_overlap 20~100
# ==============================
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

texts = splitter.split_text(text)

# ==============================
# Qdrant に書き込む
# ==============================
vector_store.add_texts(texts)

print(f"{len(texts)} 個のテキストチャンクを Qdrant collection「{COLLECTION_NAME}」に書き込みました")
