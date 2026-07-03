from nltk.translate.bleu_score import sentence_bleu


def cumulative_bleu(references, hypothesis):
    bleu_1_gram = sentence_bleu(references=references, hypothesis=hypothesis, weights=(1, 0, 0, 0))
    bleu_2_gram = sentence_bleu(references=references, hypothesis=hypothesis, weights=(0.5, 0.5, 0, 0))
    bleu_3_gram = sentence_bleu(references=references, hypothesis=hypothesis, weights=(0.33, 0.33, 0.33, 0))
    bleu_4_gram = sentence_bleu(references=references, hypothesis=hypothesis, weights=(0.25, 0.25, 0.25, 0.25))

    print('blue 1-gram:%f' % bleu_1_gram)
    print('blue 2-gram:%f' % bleu_2_gram)
    print('blue 3-gram:%f' % bleu_3_gram)
    print('blue 4-gram:%f' % bleu_4_gram)

    return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram


def test_cumulative_bleu():
    # 生成文本
    hypothesis = "This is some generated text."
    # 参考文本列表
    reference_texts = ["This is a reference text.", "This is another reference text."]

    # 计算Bleu指标

    c_bleu = cumulative_bleu(references=reference_texts, hypothesis=hypothesis)
    # 打印结果
    print("The Bleu score is:", c_bleu)


if __name__ == '__main__':
    test_cumulative_bleu()
