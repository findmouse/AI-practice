import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# グラフで日本語（SimHei）を正しく表示するための設定
plt.rcParams['font.sans-serif'] = ['SimHei']
# マイナス記号を正しく表示するための設定
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 1. DataFrameでの特定行・列データの読み込みデモ
# ---------------------------------------------------------

# 1.1 列データの読み込みデモ

# データソースを読み込み、DataFrameオブジェクトを取得
# df = pd.read_csv('data/gapminder.tsv', sep='\t')
df = pd.read_csv('data/gapminder.tsv', sep='\t')
print(df.tail())

# 2. データ処理の前に、データの構造を簡単に確認
print(df.shape)     # データの形状 (1704, 6)
print(df.dtypes)    # 各列のデータ型を確認
# df.info()         # 各列の基本情報（非欠損値の数、データ型など）を確認
print(df.describe()) # 数値列の統計情報を表示

# 特定のデータ型（int, float）の統計情報のみを表示
print(df.describe(include=['int', 'float']))

# 3. 特定の1列を読み込む
# country列を取得
print(df['country'])
# df.country でも可能

# 4. 複数の特定列を読み込む
# country, year, gdpPercap の3列を取得
print(df[['country', 'year', 'gdpPercap']])

# ---------------------------------------------------------
# 1.2 行データの読み込みデモ
# ---------------------------------------------------------

# 行インデックスを使用して特定の行を取得
# df.loc[0]   # インデックス名で取得
# df.iloc[0]  # 行番号で取得

# tail()を使用して最後の行を取得
# df.tail(n=1)
# df.tail(1)

# head()を使用して最初の行を取得
# df.head(1)

# ---------------------------------------------------------
# 1.3 特定の行と列を組み合わせて読み込む（重要）
# ---------------------------------------------------------

# 形式: df.loc[[行], [列]]   (行インデックス + 列名)
# 形式: df.iloc[[行], [列]]  (行番号 + 列番号)

# DataFrameの基本情報を出力
print(df.info())

# ilocを使用して、特定の行と列を範囲指定（range）やスライスで取得
# すべての行に対して、1列目と3列目のデータを取得
print(df.iloc[:, range(1, 5, 2)])

# 最初の10行に対して、1列目と3列目のデータを取得
print(df.iloc[range(10), range(1, 5, 2)])

# スライスを使用して取得
print(df.iloc[:, 1:5:2])

# ---------------------------------------------------------
# 2. DataFrameオブジェクト - グループ化と集計 (Grouping)
# ---------------------------------------------------------


# 元データの先頭部分を確認
print(df.head())

# グループ化統計の形式:
# df.groupby('グループ化する列')['集計対象の列'].集計関数()

# 要件1: 年ごとの平均人口と平均GDPを算出
# 年ごとの平均人口
print(df.groupby('year')['pop'].mean())
# 年ごとの平均GDP
print(df.groupby('year')['gdpPercap'].mean())