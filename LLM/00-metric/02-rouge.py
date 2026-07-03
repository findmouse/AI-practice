from rouge import Rouge

# 生成文本
hypothesis = "This is some generated text."
# 参考文本列表
reference_texts = ["This is a reference text.", "This is another reference text."]

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference_texts[1])

# print("ROUGE-1 precision:", scores[0])
print(scores)
print("ROUGE-1 recall:", scores[0]["rouge-1"]["r"])
print("ROUGE-1 precision:", scores[0]["rouge-1"]["p"])
print("ROUGE-1 score:", scores[0]["rouge-1"]["f"])