from langchain_community.document_loaders import TextLoader, UnstructuredFileLoader

# ==========================================
# 1. UnstructuredFileLoader を使って読み込む
# ==========================================
# 注意：UnstructuredFileLoader を使うには事前に依存パッケージのインストールが必要：
# pip install "unstructured[local-inference]"
try:
    loader_unstructured = UnstructuredFileLoader("./data/衣服属性.txt", encoding="utf8")
    docs_unstructured = loader_unstructured.load()

    print("--- UnstructuredFileLoader の結果 ---")
    print(docs_unstructured)
    print(f"ドキュメントチャンク数: {len(docs_unstructured)}")
    if docs_unstructured:
        print(f"先頭4文字: {docs_unstructured[0].page_content[:4]}")
except Exception as e:
    print(f"Unstructured の読み込みに失敗しました（依存関係またはサードパーティライブラリが不足している可能性があります）: {e}")

print("*" * 80)

# ==========================================
# 2. TextLoader を使って読み込む（通常のプレーンテキストに推奨）
# ==========================================
# TextLoader は LangChain で最も基本的かつ軽量なプレーンテキストローダーで、追加の依存関係が不要
loader_text = TextLoader("./data/衣服属性.txt", encoding="utf8")
docs_text = loader_text.load()

print("--- TextLoader の結果 ---")
print(docs_text)
print(f"ドキュメントチャンク数: {len(docs_text)}")
if docs_text:
    print(f"先頭4文字: {docs_text[0].page_content[:4]}")
