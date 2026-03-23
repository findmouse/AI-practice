import numpy as np
import pandas as pd
import missingno as msno
from numpy import NaN, nan, NAN
import matplotlib.pyplot as plt

# ==========================================
# 1. 欠損値の定義と判定
# ==========================================

# Pandasにおける欠損値は NaN, nan, NAN で表されます。
# これらは空文字 '', False, 0 とは異なり、「データが存在しない」状態を指します。

# 欠損値は False や 0 とは比較できません。
print(np.NaN == True)   # False
print(np.NaN == False)  # False
print(np.NaN == 0)      # False
print(np.NaN == ' ')    # False

# 欠損値は「無意味な値」であるため、NaN 同士の比較（==）も False になります。
print(np.NaN == np.nan) # False

# 欠損値の判定には Pandas の API を使用します。
# isnull() または isna(): 値が欠損している場合に True を返します。
print(pd.isnull(np.NaN)) # True
print(pd.isna(np.nan))   # True

# notnull() または notna(): 値が存在する場合に True を返します。
print(pd.notnull(10))    # True
print(pd.notna('abc'))   # True


# ==========================================
# 2. データの読み込みと欠損値の可視化
# ==========================================

# タイタニック号のデータセットを読み込み、データの概要を確認します。
df = pd.read_csv('data/titanic_train.csv')

# データの形状、基本情報、統計量を確認
print(df.shape)
df.info()
print(df.describe())

# 欠損値の分布を可視化 (missingno ライブラリを使用)
# 各カラムの欠損率をバーチャートで確認
msno.bar(df)

# カラム間の欠損値の相関（ある項目が欠損しているとき、別の項目も欠損しているか）を確認
msno.heatmap(df)
plt.show()


# ==========================================
# 3. 欠損値の処理方法
# ==========================================

# --- 方法 1: 欠損値の削除 (Deletion) ---
# 欠損値が少ない場合や、重要な項目が欠損している場合に適しています。
# subset で特定のカラムのみを基準に削除することも可能です。
df_dropped = df.dropna(how='any', subset=['Age'])

# --- 方法 2: 固定値や統計量による補完 (Imputation) ---
# 平均値(Mean)、中央値(Median)、最頻値(Mode)などを用いて補完します。
age_mean = df.Age.mean()
df['Age'].fillna(age_mean, inplace=True) # 年齢の欠損を平均値で補完

# --- 方法 3: 時系列データの補完 (Time Series Imputation) ---
# 時系列データの場合、前後の値から推測するのが一般的です。
city_day = pd.read_csv('data/city_day.csv', parse_dates=['Date'], index_col='Date')

# 前の値で補完 (Forward Fill)
# city_day['Xylene'].fillna(method='ffill')

# 後の値で補完 (Backward Fill)
# city_day['Xylene'].fillna(method='bfill')

# 線形補間 (Linear Interpolation): 前後の値から中間値を算出
# limit_direction='both' は前方・後方の両方のデータから推論します。
city_day['Xylene'].interpolate(limit_direction='both')[50:64].plot()