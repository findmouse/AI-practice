import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Optional, List

import requests

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from config import (
    LOCAL_MODEL,
    LOCAL_API_BASE,
    TEMPERATURE,
    MAX_TOKENS,
)


class QwenLLM(LLM):
    """自定义 Qwen (Ollama) LLM 类"""

    model_name: str = LOCAL_MODEL
    api_base: str = LOCAL_API_BASE
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS

    @property
    def _llm_type(self) -> str:
        return "QwenLLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """核心推理方法：调用本地 Ollama API 并返回生成的文本"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        response = requests.post(
            f"{self.api_base}/api/generate",
            json=payload,
            timeout=300,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama API 请求失败，状态码: {response.status_code}, 错误信息: {response.text}"
            )

        result = response.json()
        return result["response"]


# ==========================================
# 使用示例
# ==========================================
if __name__ == '__main__':
    llm = QwenLLM()
    prompt_text = "介绍一下Python"
    result = llm.invoke(prompt_text)

    print("=== 模型输出 ===")
    print(result)