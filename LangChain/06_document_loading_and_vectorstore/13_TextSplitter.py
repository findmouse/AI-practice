# 現代の LangChain では独立モジュール langchain_text_splitters からのインポートが推奨されている
from langchain_text_splitters import CharacterTextSplitter

# 分割器を初期化
text_splitter = CharacterTextSplitter(
    separator=" ",      # スペースを分割区切り文字とする
    chunk_size=5,       # 各チャンクの目標最大文字数
    chunk_overlap=0,    # チャンク間の重複文字数
)

# 1. 単純な文字列リストの分割、List[str] を返す
a = text_splitter.split_text("a b c d e f")
print("=== split_text の結果 ===")
print(a)
# 出力: ['a b c', 'd e f']

print("\n" + "*" * 80 + "\n")

# 2. ドキュメントリストの分割、List[Document] を返す
texts = text_splitter.create_documents(["a b c d e f", "e f g h"])
print("=== create_documents の結果 ===")
print(texts)
