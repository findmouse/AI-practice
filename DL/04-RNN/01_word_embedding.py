from janome.tokenizer import Tokenizer
import torch
import torch.nn as nn

# 形態素解析（分かち書き）
text = '東京オリンピックの進捗バーはすでに半分を過ぎ、多くの外国人選手が試合を終えて帰路についた。'
tokenizer = Tokenizer()
words = [token.surface for token in tokenizer.tokenize(text)]
# print(words)

# 重複を除去
un_words = list(set(words))
# print(un_words)
num = len(un_words)
# print(num)
# Embedding層を呼び出す
embeds = nn.Embedding(num_embeddings=num, embedding_dim=4)
# print(embeds(torch.tensor(5)))

for i, word in enumerate(un_words):
    print(word, '\t', embeds(torch.tensor(i)))
