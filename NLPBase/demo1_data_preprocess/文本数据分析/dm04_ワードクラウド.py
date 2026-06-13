# coding: utf-8
import jieba.posseg as pseg
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from itertools import chain

# GUIバックエンドの設定
matplotlib.use('TkAgg')


def extract_adjectives(text):
    """
    テキストから形容詞(けいいようし)のリストを抽出する。

    Parameters:
        text (str): 処理対象のテキスト

    Returns:
        list: 抽出された形容詞のリスト
    """
    adjective_list = []
    # 形態素解析を行い、品詞を判定
    for word, flag in pseg.lcut(text):
        # 'a' はjiebaにおける形容詞（Adjective）のタグ
        if flag == 'a':
            adjective_list.append(word)

    return adjective_list


def create_word_cloud(keywords_list):
    """
    ワードリストを基にワードクラウドを生成・表示する。

    Parameters:
        keywords_list (list): ワードクラウドの生成に使用する単語のリスト
    """
    # ワードクラウド生成器のインスタンス化（日本語フォントを指定）
    wordcloud = WordCloud(
        font_path='../../assets/SimHei.ttf',
        max_words=100,
        background_color='white'
    )

    # データの準備（単語をスペース区切りの文字列に結合）
    keyword_string = ' '.join(keywords_list)
    wordcloud.generate(keyword_string)

    # 描画処理
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # 軸を非表示
    plt.show()


def generate_word_clouds():
    """
    学習データを読み込み、ポジティブ/ネガティブデータそれぞれの形容詞ワードクラウドを生成する。
    """
    # データの読み込み（TSVファイル）
    train_data = pd.read_csv('../../assets/train.tsv', sep='\t')
    print(f'[INFO] train_data.head():\n{train_data.head()}')

    # ----------------------------------------
    # 1. ポジティブデータ (label == 1) の処理
    # ----------------------------------------
    positive_data = train_data[train_data['label'] == 1]['sentence']

    # ポジティブデータから形容詞を抽出（リストのフラット化）
    positive_adjectives = list(chain(*map(lambda x: extract_adjectives(x), positive_data)))
    print(f'[INFO] positive_adjectives count: {len(positive_adjectives)}')

    # ワードクラウドの生成
    create_word_cloud(positive_adjectives)

    # ----------------------------------------
    # 2. ネガティブデータ (label == 0) の処理
    # ----------------------------------------
    negative_data = train_data[train_data['label'] == 0]['sentence']

    # ネガティブデータから形容詞を抽出（リストのフラット化）
    negative_adjectives = list(chain(*map(lambda x: extract_adjectives(x), negative_data)))
    print(f'[INFO] negative_adjectives count: {len(negative_adjectives)}')

    # ワードクラウドの生成
    create_word_cloud(negative_adjectives)


if __name__ == '__main__':
    generate_word_clouds()