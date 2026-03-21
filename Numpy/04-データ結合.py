import numpy as np
import pandas as pd
import sqlite3

# =================================================================
# 1. データの連結 - pd.concat()
# =================================================================

# サンプルデータの作成
df1 = pd.DataFrame({'A': ['A0', 'A1'], 'B': ['B0', 'B1']}, index=[0, 1])
df2 = pd.DataFrame({'A': ['A2', 'A3'], 'B': ['B2', 'B3']}, index=[2, 3])
df3 = pd.DataFrame({'A': ['A4', 'A5'], 'B': ['B4', 'B5']}, index=[4, 5])

# 行方向の連結 (垂直結合)
result_row = pd.concat([df1, df2, df3])
print("--- 行方向の連結 ---\n", result_row)

# 列方向の連結 (水平結合)
result_col = pd.concat([df1, df2, df3], axis=1)
print("\n--- 列方向の連結 ---\n", result_col)

# インデックスをリセットして連結
result_ignore_idx = pd.concat([df1, df2, df3], ignore_index=True)

# =================================================================
# 2. データのマージ - pd.merge() (SQLライクな結合)
# =================================================================

# SQLite3を使用して実データを読み込む例 (chinook.db)
# ※ 実行環境に chinook.db が必要です
try:
    conn = sqlite3.connect('data/chinook.db')
    tracks = pd.read_sql_query("SELECT * FROM tracks", conn)
    genres = pd.read_sql_query("SELECT * FROM genres", conn)

    # 内部結合 (Inner Join): 楽曲データとジャンルデータを結合
    tracks_genres = pd.merge(tracks, genres, on='GenreId', how='inner')
    print("\n--- データのマージ (Inner Join) ---\n", tracks_genres.head())

    # 応用：ジャンルごとの平均曲時間の算出
    # MillisecondsをTimeDelta形式に変換
    tracks_genres['Duration'] = pd.to_timedelta(tracks_genres['Milliseconds'], unit='ms')
    avg_duration = tracks_genres.groupby('Name_y')['Duration'].mean()
    print("\n--- ジャンル別平均曲時間 ---\n", avg_duration)

    conn.close()
except Exception as e:
    print(f"\n[Note] DBファイルが見つからないため、マージの演習をスキップしました: {e}")

# =================================================================
# 3. インデックスによる結合 - join()
# =================================================================

# サンプルデータ
stocks_2016 = pd.DataFrame({'Symbol': ['AAPL', 'MSFT'], 'Price': [110, 60]})
stocks_2018 = pd.DataFrame({'Symbol': ['AAPL', 'GOOG'], 'Price': [170, 1000]})

# 特定の列をキーにして結合 (joinを使用する場合)
result_join = stocks_2016.join(
    stocks_2018.set_index('Symbol'),
    lsuffix='_2016',
    rsuffix='_2018',
    on='Symbol'
)
print("\n--- join() による結合 ---\n", result_join)

# -----------------------------------------------------------------
# まとめ：
# - pd.concat: 単純な連結（行または列）
# - pd.merge: 共通の列（キー）に基づく結合（SQL形式）
# - join: インデックスに基づく結合
# -----------------------------------------------------------------