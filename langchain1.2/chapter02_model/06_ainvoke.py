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


async def demo_async_invoke():
    print("=== デモ：ainvoke の非同期（ノンブロッキング）効果 ===")
    start_time = time.perf_counter()  # 開始時刻を記録
    print("プログラム開始...")
    # 1. タスク (Task) を作成
    print(">>> 非同期モデル呼び出し (ainvoke) を発行...")
    async_task = asyncio.create_task(model.ainvoke("人工知能を一言で説明してください。"))
    # 2. 他のタスクを並行して実行
    print(">>> モデルリクエストはバックグラウンドで送信済み、ローカル処理を継続...")
    for i in range(3):
        await asyncio.sleep(1)  # 非同期待機を使い、制御権を解放する
        print(f">>> {i + 1}番目のタスクを実行中... (経過時間 {time.perf_counter() - start_time:.2f}s)")
    # 3. モデルの結果を取得
    print(">>> ローカルタスク完了、モデルの状態を確認します...")
    response = await async_task
    end_time = time.perf_counter()
    print(f">>> モデルの応答: {response.content}")
    print(f"=== 総実行時間: {end_time - start_time:.2f}s ===")

async def main():
    """メイン関数"""
    await demo_async_invoke()


if __name__ == "__main__":
    asyncio.run(demo_async_invoke())

