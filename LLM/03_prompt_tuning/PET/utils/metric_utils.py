from typing import List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ClassEvaluator:
    def __init__(self):
        self.goldens = []
        self.predictions = []

    def add_batch(self, pred_batch: List, gold_batch: List):
        """1 バッチ分の予測値と正解値を追加する。"""
        if len(pred_batch) != len(gold_batch):
            raise ValueError("pred_batch と gold_batch の件数が一致しません。")
        if not gold_batch:
            return

        # 複数要素で 1 ラベルを構成する場合は文字列へ結合する。
        if isinstance(gold_batch[0], (list, tuple)):
            pred_batch = [",".join(map(str, value)) for value in pred_batch]
            gold_batch = [",".join(map(str, value)) for value in gold_batch]

        self.goldens.extend(gold_batch)
        self.predictions.extend(pred_batch)

    def compute(self, round_num=2) -> dict:
        """蓄積した値から accuracy、precision、recall、F1 を計算する。"""
        if not self.goldens:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "class_metrics": {},
            }

        classes = sorted(set(self.goldens) | set(self.predictions))
        result = {
            "accuracy": round(
                accuracy_score(self.goldens, self.predictions), round_num
            ),
            "precision": round(
                precision_score(
                    self.goldens,
                    self.predictions,
                    average="weighted",
                    zero_division=0,
                ),
                round_num,
            ),
            "recall": round(
                recall_score(
                    self.goldens,
                    self.predictions,
                    average="weighted",
                    zero_division=0,
                ),
                round_num,
            ),
            "f1": round(
                f1_score(
                    self.goldens,
                    self.predictions,
                    average="weighted",
                    zero_division=0,
                ),
                round_num,
            ),
        }

        conf_matrix = np.asarray(
            confusion_matrix(self.goldens, self.predictions, labels=classes)
        )
        class_metrics = {}
        for index, class_name in enumerate(classes):
            predicted_count = conf_matrix[:, index].sum()
            actual_count = conf_matrix[index, :].sum()
            precision = (
                0.0
                if predicted_count == 0
                else conf_matrix[index, index] / predicted_count
            )
            recall = (
                0.0
                if actual_count == 0
                else conf_matrix[index, index] / actual_count
            )
            f1 = (
                0.0
                if precision + recall == 0
                else 2 * precision * recall / (precision + recall)
            )
            class_metrics[class_name] = {
                "precision": round(float(precision), round_num),
                "recall": round(float(recall), round_num),
                "f1": round(float(f1), round_num),
            }

        result["class_metrics"] = class_metrics
        return result

    def reset(self):
        """蓄積した値をリセットする。"""
        self.goldens = []
        self.predictions = []
