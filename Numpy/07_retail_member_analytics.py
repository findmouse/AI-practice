import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os

# --- 環境設定 ---
# グラフの日本語文字化け防止（Windows: MS Gothic / Mac: AppleGothic）
plt.rcParams['font.sans-serif'] = ['MS Gothic']
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data():
    """
    データの読み込みと前処理（ローカライズおよび型変換）
    """
    excel_path = 'data/会員情報問い合わせ.xlsx'
    # 高速化のためのキャッシュファイル（Feather形式） キャッシュファイル    缓存文件
    cache_path = 'data/member_data.feather'

    # 1. キャッシュが存在する場合は高速読み込み
    if os.path.exists(cache_path):
        try:
            return pd.read_feather(cache_path)
        except Exception:
            pass  # 読み込み失敗時は通常通りExcelを読み込む

    # 2. Excelファイルの読み込み
    # 注意: 高速化のため pip install python-calamine を推奨
    try:
        df = pd.read_excel(excel_path, engine='calamine')
    except Exception:
        df = pd.read_excel(excel_path)

    # 3. カラム名の日本語化（画像の表頭に対応）
    df.columns = [
        '会員番号', '会員ランク', '入会経路', '登録時間', '所属店舗コード',
        '担当スタッフコード', '都道府県', '市区町村', '性別', '生年月日',
        '年齢', 'ライフサイクル'
    ]

    # 4. データ型の統一（混合型によるエラー防止）
    # 会員番号やスタッフコードは文字列型として扱う
    id_columns = ['会員番号', '所属店舗コード', '担当スタッフコード']
    for col in id_columns:
        df[col] = df[col].astype(str)

    # 5. 会員ランクの中身を日本語に置換
    rank_map = {
        '白银会员': 'シルバー会員',
        '黄金会员': 'ゴールド会員',
        '铂金会员': 'プラチナ会員',
        '钻石会员': 'ダイヤモンド会員'
    }
    df['会員ランク'] = df['会員ランク'].replace(rank_map)

    # 6. キャッシュの保存（次回以降の高速化）
    try:
        df.to_feather(cache_path)
    except Exception:
        pass

    return df


def analyze_monthly_trends(df):
    """
    月別の新規会員登録推移の分析
    """
    # 登録年月列の作成
    df['登録年月'] = df['登録時間'].apply(lambda x: x.strftime('%Y-%m'))

    # 月別の新規登録数を集計
    month_stats = df.groupby('登録年月')[['会員番号']].count()
    month_stats.columns = ['月間増分']

    # 会員累計数（在庫数）の計算
    month_stats['会員累計数'] = month_stats['月間増分'].cumsum()

    # 可視化：増分と累計の複合グラフ
    fig, ax = plt.subplots(figsize=(16, 8))
    month_stats['会員累計数'].plot(kind='bar', color='pink', alpha=0.7, ax=ax, label='累計会員数')
    month_stats['月間増分'][1:].plot(secondary_y=True, color='green', marker='o', ax=ax, label='月間新規登録数')

    plt.title('会員登録数の推移（月次増分および累計）', fontsize=16)
    ax.set_xlabel('登録年月')
    ax.set_ylabel('累計数')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    return month_stats


def analyze_rank_distribution(df):
    """
    会員ランク別の月次構成分析
    """
    # ピボットテーブルによる集計
    rank_pivot = df.pivot_table(
        index='登録年月', columns='会員ランク', values='会員番号', aggfunc='count'
    )

    # データの存在する期間に絞り込み
    rank_pivot = rank_pivot[1:]

    # 可視化：ランク別構成
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax2 = ax1.twinx()

    # 一般ランクと上位ランクで軸を分けて表示
    rank_pivot[['シルバー会員', 'ゴールド会員']].plot(ax=ax1, grid=True, marker='s', label=['シルバー', 'ゴールド'])
    rank_pivot[['ダイヤモンド会員', 'プラチナ会員']].plot(kind='bar', ax=ax2, alpha=0.4,
                                                          label=['ダイヤモンド', 'プラチナ'])

    ax1.set_ylabel('シルバー/ゴールド会員数')
    ax2.set_ylabel('ダイヤモンド/プラチナ会員数')
    ax2.legend(loc='upper left')
    plt.title('新規登録会員のランク別分布推移', fontsize=20)
    plt.show()


if __name__ == "__main__":
    print("データ分析を開始します...")
    try:
        # データロード
        member_df = load_and_preprocess_data()

        # 月次推移分析
        stats = analyze_monthly_trends(member_df)

        # ランク分布分析
        analyze_rank_distribution(member_df)

        print("すべての分析が正常に完了しました。")

    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
