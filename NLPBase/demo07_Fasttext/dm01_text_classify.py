import fasttext

# テキスト分類モデルの学習（初期状態）
# model = fasttext.train_supervised('./fasttext_data/cooking_train.txt')
# result1 = model.predict("Which backing dish is best to bake a banana bread ?")
# print(f"1件目のテキストの予測結果: {result1}")
#
#
# result2 = model.predict("Why not put knives in the dishwasher ?")
# print(f"2件目のテキストの予測結果: {result2}")
#
#
# # 検証データセットでの評価結果
# result3 = model.test('./fasttext_data/cooking.pre.valid')
# print(result3)


# # TODO: 【第1段階最適化】データの前処理を実施
# # テキスト分類モデルの学習
# model = fasttext.train_supervised('./fasttext_data/cooking.pre.train')
# result1 = model.predict("Which backing dish is best to bake a banana bread ?")
# print(f"1件目のテキストの予測結果: {result1}")
#
# result2 = model.predict("Why not put knives in the dishwasher ?")
# print(f"2件目のテキストの予測結果: {result2}")
#
# # 検証データセットでの評価結果
# result3 = model.test('./fasttext_data/cooking.pre.valid')
# print(result3)


# # TODO: 【第2段階最適化】Epoch数の追加
# # テキスト分類モデルの学習
# model = fasttext.train_supervised('./fasttext_data/cooking.pre.train', epoch=25)
# result1 = model.predict("Which backing dish is best to bake a banana bread ?")
# print(f"1件目のテキストの予測 key の結果: {result1}")
#
# result2 = model.predict("Why not put knives in the dishwasher ?")
# print(f"2件目のテキストの予測結果: {result2}")
#
# # 検証データセットでの評価結果
# result3 = model.test('./fasttext_data/cooking.pre.valid')
# print(result3)


# # TODO: 【第3段階最適化】学習率（lr）の引き上げ
# # テキスト分類モデルの学習
# model = fasttext.train_supervised('./fasttext_data/cooking.pre.train', epoch=25, lr=1.0)
# result1 = model.predict("Which backing dish is best to bake a banana bread ?")
# print(f"1件目のテキストの予測結果: {result1}")
#
# result2 = model.predict("Why not put knives in the dishwasher ?")
# print(f"2件目のテキストの予測結果: {result2}")
#
# # 検証データセットでの評価結果
# result3 = model.test('./fasttext_data/cooking.pre.valid')
# print(result3)


# # TODO: 【第4段階最適化】n-gram（wordNgrams）の追加
# # テキスト分類モデルの学習
# model = fasttext.train_supervised('./fasttext_data/cooking.pre.train', epoch=25, lr=1.0, wordNgrams=2)
# result1 = model.predict("Which backing dish is best to bake a banana bread ?")
# print(f"1件目のテキストの予測結果: {result1}")
#
# result2 = model.predict("Why not put knives in the dishwasher ?")
# print(f"2件目のテキストの予測結果: {result2}")
#
# # 検証データセットでの評価結果
# result3 = model.test('./fasttext_data/cooking.pre.valid')
# print(result3)


# TODO: 【第5段階最適化】損失関数（loss）の計算方式変更
# テキスト分類モデルの学習
model = fasttext.train_supervised('./fasttext_data/cooking.pre.train', epoch=25, lr=1.0, wordNgrams=2, loss="hs")
result1 = model.predict("Which backing dish is best to bake a banana bread ?")
print(f"1件目のテキストの予測結果: {result1}")

result2 = model.predict("Why not put knives in the dishwasher ?")
print(f"2件目のテキストの予測結果: {result2}")

# 検証データセットでの評価結果
result3 = model.test('./fasttext_data/cooking.pre.valid')
print(result3)

# fasttextの autotuneValidationFile 引数を使用したハイパーパラメーター自動チューニング
# TODO: 【第6段階最適化】自動チューニング（Autotune）の実行
# テキスト分類モデルの学習
model = fasttext.train_supervised(input='./fasttext_data/cooking.pre.train', autotuneDuration=60,
                                  autotuneValidationFile='./fasttext_data/cooking.pre.valid', epoch=25, lr=1.0,
                                  wordNgrams=2, loss="hs")
result1 = model.predict("Which backing dish is best to bake a banana bread ?")
print(f"1件目のテキストの予測結果: {result1}")

result2 = model.predict("Why not put knives in the dishwasher ?")
print(f"2件目のテキストの予測結果: {result2}")

# 検証データセットでの評価結果
result3 = model.test('./fasttext_data/cooking.pre.valid')
print(result3)