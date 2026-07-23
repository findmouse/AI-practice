import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from train import patch_chatglm_tokenizer
from data_handle.data_preprocess import encode_text, convert_example


def main():
    tok = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    tok = patch_chatglm_tokenizer(tok)

    text = "你好世界"
    ids = encode_text(tok, text)
    print("encode_text ok", ids[:10], "len", len(ids))

    ids2 = tok.encode(text, add_special_tokens=False)
    print("tok.encode ok", ids2[:10], "len", len(ids2))

    sample = {
        "context": "Instruction: test\nAnswer: ",
        "target": '["ok"]',
    }
    ex = {"text": [json.dumps(sample, ensure_ascii=False)]}
    out = convert_example(ex, tok, max_source_seq_len=64, max_target_seq_len=32)
    print("convert ok", out["input_ids"].shape, out["labels"].shape)
    print("smoke test passed")


if __name__ == "__main__":
    main()
