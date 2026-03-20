# --- Section 1 ---
# ライブラリのインポート
import numpy as np
import pandas as pd

# 1. データソースを読み込み、DataFrameオブジェクトを作成
df = pd.read_csv('data/scientists.csv')

# 2. DataFrameの主な属性を確認
print(df.ndim)    # 次数（次元数）を表示
print(df.shape)   # 形状（行数、列数）を表示
print(df.size)    # 要素数（欠損値NaNを含む総セル数）を表示
print(df.index)   # インデックス（行ラベル）を取得
print(df.columns) # カラム名（列ラベル）を取得
print(df.values)  # データを要素（値）のみで取得

# --- Section 2 ---
print(len(df))    # 行数を取得
df.head()         # 先頭5行を表示（デフォルト）
df.head(n=3)      # 先頭3行を表示
df.tail()         # 末尾5行を表示
df.keys()         # 全カラム名を取得（df.columnsと同等）
df.info()         # 各列のデータ型や非欠損値数などの基本情報を確認
df.describe()     # 数値型の列に対する統計情報（平均、標準偏差など）を確認
df.mean()         # 各列の平均値を算出
df.count()        # 各列の非欠損値の数をカウント

# --- Section 3 ---
# 要件: movie.csvにおいて、上映時間が平均より長い映画情報を抽出
movie_df = pd.read_csv('data/movie.csv')
# 条件に合致するデータ（duration > 平均値）を取得
print(movie_df[movie_df.duration > movie_df.duration.mean()])

# --- Section 4 ---
# DataFrameと数値の演算（各要素に対して演算が適用される）
print(df * 2)

# --- Section 7 ---
# データの出力（エクスポート）
# CSV形式で保存
df.to_csv('./output/scientists_noindex.csv', index=False)
# TSV形式（タブ区切り）で保存
df.to_csv('./output/scientists_noindex.tsv', sep='\t', index=False)
# Excel形式で保存
df.to_excel('./output/scientists.xlsx', index=False)

# データの読み込み（インポート）
pd.read_excel('./output/scientists.xlsx') # Excelファイルの読み込み
pd.read_csv('./output/scientists_noindex.csv') # CSVファイルの読み込み