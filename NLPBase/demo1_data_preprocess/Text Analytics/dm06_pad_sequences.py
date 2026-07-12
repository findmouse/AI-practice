from keras.utils import pad_sequences

cutlen = 10


def padding(x_train):
    """
    入力テキストテンソルの長さを統一する。

    :param x_train: テキストのテンソル表現。形式: [[1, 32, 32, 61], [2, 54, 21, 7, 19]]
    :return: パディングおよび切り捨て処理後のテキストテンソル表現
    """
    # pad_sequences を使用して長さを統一（Keras 3: keras.utils.pad_sequences）
    return pad_sequences(x_train, maxlen=cutlen, padding="pre", truncating="post")


if __name__ == '__main__':
    data = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1], [2, 32, 1, 23, 1]]
    my_data = padding(data)
    print(my_data)
