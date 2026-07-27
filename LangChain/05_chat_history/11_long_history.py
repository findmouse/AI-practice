from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import messages_from_dict, messages_to_dict

# 1. メッセージ履歴管理オブジェクトをインスタンス化
history = ChatMessageHistory()

# 2. 対話メッセージを追加
history.add_user_message("hi!")
history.add_ai_message("whats up?")

# 3. メッセージリストを Python の辞書リスト (Dicts) にシリアライズ
dicts = messages_to_dict(history.messages)
print("--- Dicts にシリアライズ ---")
print(dicts)

print("\n" + "*" * 80 + "\n")

# 4. 辞書リストから Message オブジェクトのリストにデシリアライズして復元
new_messages = messages_from_dict(dicts)
print("--- Message オブジェクトにデシリアライズして復元 ---")
print(new_messages)
