from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
import asyncio
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


async def demo_async_stream():
    """非同期呼び出しのノンブロッキングな挙動をデモする"""
    print("=== デモ：astream の非同期（ノンブロッキング）効果 ===")
    start_time = time.perf_counter()  # 開始時刻を記録
    print("プログラム開始...")
    # 1. 非同期ストリーミングリクエストを発行
    # 注意：この時点でリクエストはすでに送信済みで、返るのは非同期ジェネレータ
    print(">>> 非同期ストリーミング呼び出し (astream) を発行...")
    stream_resp = model.astream("機械学習の基本概念を一言で説明してください。")
    # 2. ストリーミング応答を待つ間、他のタスクを実行
    print(">>> ストリーミングリクエストは送信済み、プログラムは待たずに他の非同期タスクを継続...")
    for i in range(3):
        # time.sleep ではなく asyncio.sleep を使用
        # これにより、待機中にイベントループが上記の stream_resp のネットワーク I/O を処理できる
        await asyncio.sleep(1)
        # print(f">>> 並行タスク {i + 1} を実行中... ")
        print(f">>> {i + 1}番目のタスクを実行中... (経過時間 {time.perf_counter() - start_time:.2f}s)")
    # 3. ここからストリーミング結果の処理を開始
    print(">>> 模擬タスクが完了、バッファ内のストリーミング結果の読み取りを開始します...")
    end_time = time.perf_counter()
    print(">>> ストリーミング出力: ", end="", flush=True)
    async for chunk in stream_resp:
        # LangChain のメッセージチャンクは通常 .content で内容を取得する
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        print(content, end="", flush=True)
    print("\n>>> ストリーミング出力終了\n")
    print(f"=== 総実行時間: {end_time - start_time:.2f}s ===")


async def main():
    """メイン関数"""
    await demo_async_stream()


if __name__ == "__main__":
    asyncio.run(main())
