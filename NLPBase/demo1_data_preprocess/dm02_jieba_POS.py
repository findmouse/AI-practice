from jieba import posseg as pseg

# 文章を分かち書きし、品詞解析を行う
content = '我爱北京天安门'

result = pseg.lcut(content)
print(result)
