# N-gramの範囲（2の場合はBigram）
N_GRAM = 2


def generate_ngram(input_list: list):
    """
    入力されたリストからN-gram（連続する要素の組み合わせ）を生成する。

    Parameters:
        input_list (list): 処理対象の要素リスト
    """
    # インデックスをずらしたリストのリストを作成
    # 例: [[1, 3, 2, ...], [3, 2, 1, ...]]
    slided_lists = [input_list[i:] for i in range(N_GRAM)]
    print(f"[DEBUG] slided_lists: {slided_lists}")

    # zipで各リストの要素をペアリングし、重複を排除するためセット(set)に変換
    ngram_results = set(zip(*slided_lists))
    print(f"[DEBUG] ngram_results: {ngram_results}")


if __name__ == '__main__':
    sample_list = [1, 3, 2, 1, 5, 3]
    generate_ngram(sample_list)