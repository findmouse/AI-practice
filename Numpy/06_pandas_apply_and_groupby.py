"""
Pandas データ処理の実践: apply関数とグループ化統計
このスクリプトは、データ分析におけるカスタム関数の適用(apply)と、
高度なグループ化(groupby)操作の学習プロセスをまとめたものです。
"""

import numpy as np
import pandas as pd
import os

# --- 0. 環境設定 ---
# 実行環境に合わせてカレントディレクトリを変更
os.chdir('C:/workspace/Ai-practice/Numpy')
print(f"現在のワークディレクトリ: {os.getcwd()}")

# --- 1. apply() 関数の活用 ---
print("--- 1. apply() 関数の活用 ---")

# サンプルデータの作成
df = pd.DataFrame({'a': [10, 20, 30], 'b': [20, 30, 40]})


# 1.1 Seriesオブジェクトへの適用
# 各要素を2乗するカスタム関数
def my_func1(x):
    return x ** 2


print("Seriesへのapply適用結果:")
print(df['a'].apply(my_func1))


# 1.2 引数を伴う関数の適用
def my_func2(x, e):
    return x ** e


# パラメーターを設定
print("引数(e=3)を指定した適用結果:")
print(df['a'].apply(my_func2, e=3))


# 1.3 DataFrameオブジェクトへの適用 (axisの制御)
def check_type(x):
    # axis=0なら列(Series)、axis=1なら行(Series)が渡される
    return type(x)


print("DataFrameへの適用 (行単位):")
print(df.apply(check_type, axis=1))

# --- 2. 実践ケーススタディ: タイタニック・データセット ---
print("\n--- 2. 実践: タイタニックデータの欠損値分析 ---")


# 本来はファイルから読み込みますが、ここでは構造のデモのみ記述
# titanic = pd.read_csv('data/titanic_train.csv')

def count_missing(vec):
    """欠損値の個数をカウント"""
    return pd.isnull(vec).sum()


def prop_missing(vec):
    """欠損値の割合を算出"""
    return count_missing(vec) / vec.size


def prop_complete(vec):
    """有効値（非欠損値）の割合を算出"""
    return 1 - prop_missing(vec)


# 列単位での一括統計（デモ用実行コードはコメントアウト）
# print(titanic.apply(count_missing))

# --- 3. 関数のベクトル化 (np.vectorize) ---
print("\n--- 3. 関数のベクトル化 ---")


@np.vectorize
def avg_2_mod(x, y):
    """
    特定の条件（x=20）でNaNを返し、それ以外は平均を計算する関数。
    通常のPython関数をベクトル化し、PandasのSeries同士の演算を可能にする。
    """
    if x == 20:
        return np.NaN
    else:
        return (x + y) / 2


print("ベクトル化関数の実行結果:")
print(avg_2_mod(df['a'], df['b']))

# --- 4. グループ化操作 (Groupby) ---
print("\n--- 4. グループ化と統計処理 ---")


# 世界各国の統計データ(Gapminder)を想定
df_gap = pd.read_csv('data/gapminder.tsv', sep='\t')

# 4.1 データの集約 (Aggregation)
# 期待値: 大陸別の平均寿命を算出
result = df_gap.groupby('continent')['lifeExp'].agg([np.mean, np.std])

# 4.2 データの変換 (Transformation)
# Zスコア（標準化）の計算: (x - 平均) / 標準偏差
def my_zscore(x):
    return (x - x.mean()) / x.std()


# 年ごとのグループ内で標準化を実施
df_gap['lifeExp_z'] = df_gap.groupby('year')['lifeExp'].transform(my_zscore)

# 4.3 データのフィルタリング (Filter)
# レストランのチップデータを用い、利用回数が10回以上のグループ（人数）のみ抽出
tips = pd.read_csv('data/tips.csv')
tips_filtered = tips.groupby('size').filter(lambda x: x['size'].count() > 10)

print("処理完了。詳細は各メソッドのコメントを参照してください。")
