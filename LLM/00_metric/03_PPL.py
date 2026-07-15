import math
from fractions import Fraction

# コーパスの定義
sentences = [
    ['I', 'have', 'a', 'pen'],
    ['He', 'has', 'a', 'book'],
    ['She', 'has', 'a', 'cat']
]
# 言語モデルの定義
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

# パープレキシティの計算
perplexity = 0
for sentence in sentences:
    sentence_prob = 1
    for word in sentence:
        sentence_prob *= unigram[word]
    temp = -math.log(sentence_prob, 2) / len(sentence)
    perplexity += 2 ** temp

perplexity = perplexity / len(sentences)
print(f"パープレキシティ: {perplexity}")