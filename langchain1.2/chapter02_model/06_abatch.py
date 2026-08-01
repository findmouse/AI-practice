import asyncio
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import time

# .envファイルから環境変数を読み込む
load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")

model = init_chat_model(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


async def demo_async_batch(questions):
    """abatch の非同期（ノンブロッキング）な挙動をデモする"""
    print("=== デモ：abatch の非同期（ノンブロッキング）効果 ===")
    start_time = time.perf_counter()  # 開始時刻を記録
    print("プログラム開始...")
    # バッチ入力を準備
    # questions = ["深層学習と従来の機械学習の違いを一言で説明してください", "中国の首都はどこですか？"]
    # 1. 非同期バッチリクエストを発行
    # ポイント：create_task を使うことでコルーチンを即座にバックグラウンドで実行させる
    print(">>> 非同期バッチ呼び出し (abatch) を発行...")
    batch_task = asyncio.create_task(model.abatch(questions))
    # 2. バッチ処理を待つ間、他のタスクを実行
    print(">>> バッチタスクはバックグラウンドで実行中、メインプログラムは処理を継続...")
    for i in range(3):
        # ポイント：asyncio.sleep を使うことでバックグラウンドタスクがネットワークリクエストのために CPU 時間を確保できる
        await asyncio.sleep(1)
        print(f">>> {i + 1}番目のタスクを実行中... (経過時間 {time.perf_counter() - start_time:.2f}s)")
    # 3. バッチ処理の結果を待つ
    print(">>> 他のタスクが完了、バックグラウンドのバッチタスクの結果を取得します...")
    # この時点で batch_task はすでに完了しているか、ここで完了を待つことになる
    responses = await batch_task
    end_time = time.perf_counter()
    for response in responses:
        content = response.content if hasattr(response, 'content') else str(response)
        print(f">>> レスポンス内容: {content}")
    print(f"=== 総実行時間: {end_time - start_time:.2f}s ===")


async def main():
    """メイン関数"""

    questions = ["深層学習と従来の機械学習の違いを一言で説明してください", "中国の首都はどこですか？"]
    await demo_async_batch(questions)
    print("questions1 完了")
    questions2 = ["確率的勾配降下法を一言で説明してください", "日本の首都はどこですか？"]
    await demo_async_batch(questions2)
    print("questions2 完了")


if __name__ == "__main__":
    asyncio.run(main())
    print("__main__完了")
