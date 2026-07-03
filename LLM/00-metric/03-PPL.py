import math
from fractions import Fraction

# 定义语料库
sentences = [
    ['I', 'have', 'a', 'pen'],
    ['He', 'has', 'a', 'book'],
    ['She', 'has', 'a', 'cat']
]
# 定义语言模型
unigram = {}
unigram_sum = 0
for i in sentences:
    for j in i:
        if j not in unigram.keys():
            unigram[j] = 1
        else:
            unigram[j] += 1
        unigram_sum += 1

for k in unigram.keys():
    unigram[k] = Fraction(unigram[k], unigram_sum)
print(unigram)

# 计算困惑度
perplexity = 0
for sentence in sentences:
    sentence_prob = 1
    for word in sentence:
        sentence_prob *= unigram[word]
    temp = -math.log(sentence_prob, 2) / len(sentence)
    perplexity += 2 ** temp

perplexity = perplexity / len(sentences)
print(f"困惑度为：{perplexity}")
