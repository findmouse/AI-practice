import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.indexing import index
from langchain_text_splitters import CharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import OPENAI_API_KEY, QDRANT_URL
from record_manager import SQLiteRecordManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
COLLECTION_NAME = "pku"
SOURCE_PATH = os.path.join(BASE_DIR, "..", "06_document_loading_and_vectorstore", "data", "pku.txt")

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
# テキストを読み込んで分割
# chunk_overlap 20~100
# ==============================
with open(SOURCE_PATH, encoding="utf-8") as f:
    text = f.read()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

docs = [
    Document(page_content=chunk, metadata={"source": SOURCE_PATH})
    for chunk in splitter.split_text(text)
]

# ==============================
# インデックス管理（LangChain の index() API）
# RecordManager がチャンクのハッシュを記録しており、再実行しても：
#   - 内容が変わっていないチャンクは再埋め込みしない（OpenAI API 呼び出し節約）
#   - pku.txt から消えたチャンクは Qdrant から自動的に削除される（cleanup="full"）
# ==============================
record_manager = SQLiteRecordManager(
    namespace=f"qdrant/{COLLECTION_NAME}",
    db_path=os.path.join(BASE_DIR, "record_manager.db"),
)
record_manager.create_schema()

result = index(
    docs,
    record_manager,
    vector_store,
    cleanup="full",
    source_id_key="source",
)

print(f"インデックス結果: {result}")
