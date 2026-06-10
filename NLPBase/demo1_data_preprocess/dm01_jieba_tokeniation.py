# coding:utf-8

import jieba

# 分割対象のテキストサンプル（中国語簡体字）
content1 = "我是小张张，我很喜欢动漫，以前就很喜欢看，我看的第一部动漫是海贼王"
# 中国語繁體
content2 = "我是小张张，我很喜歡動漫，以前就很喜歡看，我看的第一部動漫是海賊王"
content = content2


# TODO: 精密モード（Accurate Mode）分詞、イテレータ（iterator）を返す
def demo1_jieba_cut():
    # cutメソッド、精密モード（デフォルトもFalse）
    result1 = jieba.cut(content, cut_all=False)
    for token in result1:
        print(token)


# TODO: 精密モード（Accurate Mode）分詞、リスト（list）を返す
def demo1_jieba_lcut():
    # lcutメソッド、精密モード
    result1 = jieba.lcut(content, cut_all=False)
    print(result1)


# TODO: 全モード（Full Mode）分詞、イテレータ（iterator）を返す
def demo2_jieba_cut():
    # cutメソッド、全モード
    result1 = jieba.cut(content, cut_all=True)
    for token in result1:
        print(token)


# TODO: 全モード（Full Mode）分詞、リスト（list）を返す
def demo2_jieba_lcut():
    # lcutメソッド、全モード
    result1 = jieba.lcut(content, cut_all=True)
    print(result1)


# TODO: 検索エンジンモード（Search Engine Mode）分詞、イテレータ（iterator）を返す
def demo3_jieba_cut_for_search():
    # cut_for_searchメソッド
    result1 = jieba.cut_for_search(content)
    # イテレータをリストに変換して出力
    print(list(result1))


# TODO: 検索エンジンモード（Search Engine Mode）分詞、リスト（list）を返す
def demo3_jieba_lcut_for_search():
    # lcut_for_searchメソッド
    result1 = jieba.lcut_for_search(content)
    print(result1)

# TODO: ユーザー辞書
def demo4_jieba_custom_dict():
    jieba.load_userdict("../assets/userdict.txt")
    # cutメソッド、精密モード（デフォルトもFalse）
    result1 = jieba.lcut(content, cut_all=False)
    print(result1)



if __name__ == '__main__':
    # print("--- 1. 精密モード (lcut) ---")
    # demo1_jieba_lcut()
    #
    # print("\n--- 2. 全モード (lcut) ---")
    # demo2_jieba_lcut()
    #
    # print("\n--- 3. 検索エンジンモード (lcut) ---")
    # demo3_jieba_lcut_for_search()


    # ユーザー辞書
    demo1_jieba_lcut()
    demo4_jieba_custom_dict()
