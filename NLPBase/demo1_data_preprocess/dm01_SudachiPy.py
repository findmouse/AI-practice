from sudachipy import Dictionary, Tokenizer

tokenizer = Dictionary().create()

# 動词の活用（変形）が含まれる文章に変更
# 買った（原形：買う）、読んでいる（原形：読む）
text = "本を買ったし、小説も読んでいる。"

print("--- Mode A 形態素解析結果 ---")
tokens = tokenizer.tokenize(text, Tokenizer.SplitMode.A)
words = [t.surface() for t in tokens]
print(words)

print("\n--- 動詞の原形（辞書形）復元テスト ---")
for token in tokens:
    # 品詞の第一項目が「動詞」のものだけを抽出
    # print(token.part_of_speech()[0])
    # print(type(token))
    if token.part_of_speech()[0] == "動詞":
        print(f"表面形 (surface): {token.surface():<5} -> 辞書形 (dictionary_form): {token.dictionary_form()}")
