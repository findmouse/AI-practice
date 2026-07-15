from rouge import Rouge

# 生成テキスト
hypothesis = "This is some generated text."
# 参照テキストのリスト
reference_texts = ["This is a reference text.", "This is another reference text."]

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference_texts[1])

# print("ROUGE-1 precision:", scores[0])
print(scores)
print("ROUGE-1 再現率 (Recall):", scores[0]["rouge-1"]["r"])
print("ROUGE-1 適合率 (Precision):", scores[0]["rouge-1"]["p"])
print("ROUGE-1 F値 (F-score):", scores[0]["rouge-1"]["f"])