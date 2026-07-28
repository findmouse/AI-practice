import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any, List, Optional
import requests
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from config import (DEEPSEEK_API_KEY,
                    DEEPSEEK_API_BASE,
                    DEEPSEEK_MODEL,
                    TEMPERATURE,
                    MAX_TOKENS,
                    )


class LocalModel(LLM):
    """自定义 DeepSeek LLM 类"""

    api_key: str = DEEPSEEK_API_KEY
    api_base: str = DEEPSEEK_API_BASE
    model_name: str = DEEPSEEK_MODEL
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS

    @property
    def _llm_type(self) -> str:
        return "DeepSeek"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """核心推理方法：调用 DeepSeek API 并返回生成的文本"""
        api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError(
                "请提供 DEEPSEEK_API_KEY，或在环境变量中设置 DEEPSEEK_API_KEY"
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if stop:
            payload["stop"] = stop

        # 发送 API 请求
        response = requests.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"DeepSeek API 请求失败，状态码: {response.status_code}, 错误信息: {response.text}"
            )

        res_json = response.json()
        return res_json["choices"][0]["message"]["content"]


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    llm = LocalModel()
    prompt_text = "Pythonで深いコピーと浅コピーの区別は？"
    result = llm.invoke(prompt_text)

    print("=== 模型输出 ===")
    print(result)
